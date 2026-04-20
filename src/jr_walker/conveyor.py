"""Conveyor-belt order fulfillment phase.

Each robot orbits its assigned nest, picking SKUs it needs for the current order,
then fulfilling at the near end before looping back.

Full track (far-corner entry):
  Outer pass east → far corner → Line B west → fulfill → return path

Gap-shortcut tracks:
  Outer pass east up to gap → drop through gap → Line B west → fulfill → return path

On each new order the robot chooses the leftmost gap whose position is strictly
beyond the rightmost item it still needs, skipping unnecessary travel.
"""
from __future__ import annotations

import datetime
import time
from typing import Dict, List, Optional, Tuple

from jr_walker.entities import ActionEntry, JobKind, NestConfig, Order, Robot
from jr_walker.pathfinder import find_path, manhattan
from jr_walker.warehouse import SpacetimeGrid


# ---------------------------------------------------------------------------
# Track builder
# ---------------------------------------------------------------------------

def build_conveyor_track(nest_config: NestConfig) -> List[Tuple[int, int]]:
    """Full-orbit track: outer pass → far corner → Line B → fulfill → return."""
    nx, ny = nest_config.anchor
    n = nest_config.n_positions
    track: List[Tuple[int, int]] = []

    # Outer pass: east along y=ny+3 (picks from Line C at y=ny+2)
    for x in range(nx, nx + n + 2):
        track.append((x, ny + 3))

    # Corner: step inward from ny+3 to ny+1
    track.append((nx + n + 1, ny + 2))
    track.append((nx + n + 1, ny + 1))

    # Line B: west along y=ny+1 (picks from Line A at y=ny)
    for x in range(nx + n, nx - 2, -1):
        track.append((x, ny + 1))

    # Fulfill at near end
    track.append((nx - 1, ny))

    # Return path
    track.append((nx - 2, ny))
    track.append((nx - 2, ny + 1))
    track.append((nx - 2, ny + 2))
    track.append((nx - 2, ny + 3))
    track.append((nx - 1, ny + 3))

    return track


def _build_gap_track(nest_config: NestConfig, gap_idx: int) -> List[Tuple[int, int]]:
    """Shortcut track: outer pass to gap → drop through gap → Line B → fulfill → return."""
    nx, ny = nest_config.anchor
    track: List[Tuple[int, int]] = []

    # Outer pass east up to and including the gap position
    for x in range(nx, nx + gap_idx + 1):
        track.append((x, ny + 3))

    # Drop through the gap (empty Line C slot at ny+2, then into Line B at ny+1)
    track.append((nx + gap_idx, ny + 2))
    track.append((nx + gap_idx, ny + 1))

    # Line B west from (gap_idx - 1) down to fulfill_near
    for x in range(nx + gap_idx - 1, nx - 2, -1):
        track.append((x, ny + 1))

    # Fulfill at near end
    track.append((nx - 1, ny))

    # Return path (identical to full track)
    track.append((nx - 2, ny))
    track.append((nx - 2, ny + 1))
    track.append((nx - 2, ny + 2))
    track.append((nx - 2, ny + 3))
    track.append((nx - 1, ny + 3))

    return track


def _choose_entrance_gap(order: Order, nest_config: NestConfig) -> Optional[int]:
    """Return the leftmost gap index that still covers all items the order needs.

    A gap at index g is valid when every needed item (Line A or Line C) is at an
    index strictly less than g, so the outer pass visits all needed Line C items
    and the Line B descent from g visits all needed Line A items.
    Returns None if no gap saves travel (must use the full far-corner track).
    """
    line_a = nest_config.line_a_pallets
    line_c = nest_config.line_c_pallets

    max_needed_idx = -1
    for i, sku in enumerate(line_a):
        if sku != 0 and order.items.get(sku, 0) > 0:
            max_needed_idx = max(max_needed_idx, i)
    for i, sku in enumerate(line_c):
        if sku != 0 and order.items.get(sku, 0) > 0:
            max_needed_idx = max(max_needed_idx, i)

    # Pick leftmost gap strictly beyond the last needed position
    for i, sku in enumerate(line_c):
        if sku == 0 and i > max_needed_idx:
            return i
    return None


def pallet_to_pick_from(
    track_x: int,
    track_y: int,
    nest_config: NestConfig,
) -> Optional[Tuple[int, int]]:
    """Return the nest pallet cell to pick from when at this track position, if any."""
    nx, ny = nest_config.anchor
    n = nest_config.n_positions
    pos_idx = track_x - nx

    # Outer pass (y=ny+3): pick from Line C (y=ny+2) — but only non-gap positions
    if track_y == ny + 3 and 0 <= pos_idx < n:
        if nest_config.line_c_pallets[pos_idx] != 0:
            return (track_x, ny + 2)

    # Line B (y=ny+1): pick from Line A (y=ny) — only filled positions
    if track_y == ny + 1 and 0 <= pos_idx < n:
        if nest_config.line_a_pallets[pos_idx] != 0:
            return (track_x, ny)

    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plan_order_phase(
    nest_config: NestConfig,
    nest_robots: List[Robot],
    orders: List[Order],
    grid: SpacetimeGrid,
    actions: List[ActionEntry],
    other_nest_rects: List[Tuple[int, int, int, int]],
    strict_no_swap: bool,
    stall_limit: int,
    max_idle_ticks: int,
    max_ticks: int,
    start_time: float,
    max_runtime_seconds: float,
    progress_order_interval: int,
    progress_tick_interval: int,
) -> List[Order]:
    """Assign and plan all orders for this nest's robots. Returns the fulfilled orders."""
    nx, ny = nest_config.anchor
    full_track = build_conveyor_track(nest_config)

    # Pre-build a shortcut track for every gap in Line C
    gap_tracks: Dict[int, List[Tuple[int, int]]] = {
        i: _build_gap_track(nest_config, i)
        for i, sku in enumerate(nest_config.line_c_pallets)
        if sku == 0
    }
    fulfill_near = nest_config.fulfill_near

    def _track_for_order(order: Order) -> List[Tuple[int, int]]:
        gap = _choose_entrance_gap(order, nest_config)
        return gap_tracks[gap] if gap is not None else full_track

    def _fulfill_idx_in_track(t: List[Tuple[int, int]]) -> int:
        return next(i for i, p in enumerate(t) if p == fulfill_near)

    # Build SKU→position map from config (source of truth for what was placed where)
    sku_to_pos: Dict[int, Tuple[int, int]] = {}
    for i, sku in enumerate(nest_config.line_a_pallets):
        if sku != 0:
            sku_to_pos[sku] = (nx + i, ny)
    for i, sku in enumerate(nest_config.line_c_pallets):
        if sku != 0:
            sku_to_pos[sku] = (nx + i, ny + 2)

    pending_orders = sorted(
        [o for o in orders if not o.is_fulfilled],
        key=lambda o: o.total_picks,
    )
    fulfilled: List[Order] = []

    sorted_robots = sorted(nest_robots, key=lambda r: manhattan(r.x, r.y, nx, ny + 3))

    # Spread robots evenly around the full track for initial positioning.
    num_robots = len(sorted_robots)
    spread = len(full_track) // max(num_robots, 1)

    for r in sorted_robots:
        grid.hold_for(r.last_tick + 1, r.x, r.y)

    robot_start_idx: Dict[int, int] = {}
    for i, robot in enumerate(sorted_robots):
        start_idx = (i * spread) % len(full_track)
        robot_start_idx[robot.id] = start_idx
        tx, ty = full_track[start_idx]
        grid.release_hold(robot.last_tick + 1, robot.x, robot.y)
        if (robot.x, robot.y) != (tx, ty):
            _navigate_to_position(robot, tx, ty, grid, actions, other_nest_rects, strict_no_swap)
        grid.hold_for(robot.last_tick + 1, robot.x, robot.y)

    for r in sorted_robots:
        grid.release_hold(r.last_tick + 1, r.x, r.y)

    order_idx = 0
    robot_order_map: Dict[int, Optional[Order]] = {r.id: None for r in sorted_robots}
    # Each robot has its own current track (optimized per order) and index into it
    robot_cur_track: Dict[int, List[Tuple[int, int]]] = {r.id: full_track for r in sorted_robots}
    robot_track_idx: Dict[int, int] = {r.id: robot_start_idx[r.id] for r in sorted_robots}

    for robot in sorted_robots:
        if order_idx < len(pending_orders):
            o = pending_orders[order_idx]
            o.assigned_tick = robot.last_tick
            robot_order_map[robot.id] = o
            order_idx += 1

    tick = max(r.last_tick for r in sorted_robots)
    last_progress_tick = 0
    last_progress_orders = 0

    while True:
        if all(robot_order_map.get(r.id) is None for r in sorted_robots):
            for r in sorted_robots:
                r.job = JobKind.DONE
            break

        tick += 1
        if tick >= max_ticks:
            raise RuntimeError(f"Exceeded max_ticks={max_ticks} during order phase")
        if time.time() - start_time > max_runtime_seconds:
            raise RuntimeError("Exceeded max_runtime during order phase")

        robot_positions: Dict[int, Tuple[int, int]] = {r.id: (r.x, r.y) for r in sorted_robots}

        for robot in sorted_robots:
            if robot.last_tick >= tick:
                continue

            order = robot_order_map.get(robot.id)
            cur_track = robot_cur_track[robot.id]
            track_pos = robot_track_idx[robot.id]
            cur_x, cur_y = cur_track[track_pos % len(cur_track)]
            next_idx = (track_pos + 1) % len(cur_track)
            next_x, next_y = cur_track[next_idx]

            if order is None:
                # Done robot: keep orbiting so it doesn't block the track for others.
                collision = any(
                    rid != robot.id and pos == (next_x, next_y)
                    for rid, pos in robot_positions.items()
                )
                if grid.is_free(tick, next_x, next_y) and not collision:
                    actions.append(ActionEntry(tick, robot.id, "move", next_x, next_y))
                    grid.reserve(tick, next_x, next_y)
                    robot.x, robot.y = next_x, next_y
                    robot.last_tick = tick
                    robot_track_idx[robot.id] = next_idx
                    robot_positions[robot.id] = (next_x, next_y)
                continue

            # Pick from adjacent pallet if needed
            pallet_pos = pallet_to_pick_from(cur_x, cur_y, nest_config)
            if pallet_pos is not None:
                pallet_sku = next(
                    (sku for sku, pos in sku_to_pos.items() if pos == pallet_pos), None
                )
                if pallet_sku is not None:
                    needed = order.items.get(pallet_sku, 0) - robot.inventory.get(pallet_sku, 0)
                    if needed > 0:
                        for _ in range(needed):
                            actions.append(ActionEntry(tick, robot.id, "pick",
                                                       pallet_pos[0], pallet_pos[1]))
                            robot.inventory[pallet_sku] += 1
                            tick += 1
                        robot.last_tick = tick - 1
                        continue

            inventory_complete = not any(
                order.items.get(s, 0) > robot.inventory.get(s, 0)
                for s in order.items
            )
            if (cur_x, cur_y) == fulfill_near and inventory_complete:
                _verify_inventory(robot, order)
                actions.append(ActionEntry(tick, robot.id, "fulfill", cur_x, cur_y))
                order.fulfilled_tick = tick
                fulfilled.append(order)
                robot.inventory.clear()
                robot.last_tick = tick
                tick += 1

                if order_idx < len(pending_orders):
                    next_order = pending_orders[order_idx]
                    next_order.assigned_tick = tick
                    robot_order_map[robot.id] = next_order
                    order_idx += 1
                    # Switch to the optimal track for the new order
                    new_track = _track_for_order(next_order)
                    robot_cur_track[robot.id] = new_track
                    robot_track_idx[robot.id] = _fulfill_idx_in_track(new_track)
                else:
                    robot_order_map[robot.id] = None

                if (len(fulfilled) - last_progress_orders >= progress_order_interval
                        or tick - last_progress_tick >= progress_tick_interval):
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}] tick={tick}  fulfilled={len(fulfilled)}")
                    last_progress_orders = len(fulfilled)
                    last_progress_tick = tick
                continue

            # Move to next track position
            collision = any(
                rid != robot.id and pos == (next_x, next_y)
                for rid, pos in robot_positions.items()
            )
            if grid.is_free(tick, next_x, next_y) and not collision:
                actions.append(ActionEntry(tick, robot.id, "move", next_x, next_y))
                grid.reserve(tick, next_x, next_y)
                robot.x, robot.y = next_x, next_y
                robot.last_tick = tick
                robot_track_idx[robot.id] = next_idx
                robot_positions[robot.id] = (next_x, next_y)
            else:
                _check_stall(robot, tick, stall_limit, actions)

    return fulfilled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _navigate_to_position(
    robot: Robot,
    goal_x: int,
    goal_y: int,
    grid: SpacetimeGrid,
    actions: List[ActionEntry],
    ex_rects: List[Tuple[int, int, int, int]],
    strict_no_swap: bool,
) -> None:
    path = find_path(robot.x, robot.y, robot.last_tick, goal_x, goal_y, grid,
                     excluded_rects=ex_rects, strict_no_swap=strict_no_swap)
    if not path:
        raise RuntimeError(
            f"Robot {robot.id} cannot reach track position ({goal_x},{goal_y}) "
            f"from ({robot.x},{robot.y}) at t={robot.last_tick}")
    for i, (t, x, y) in enumerate(path):
        if i == 0:
            continue
        _, px, py = path[i - 1]
        if x != px or y != py:
            actions.append(ActionEntry(t, robot.id, "move", x, y))
        grid.reserve(t, x, y)
    if path:
        _, lx, ly = path[-1]
        robot.x, robot.y = lx, ly
        robot.last_tick = path[-1][0]
        robot.last_tick = path[-1][0]


def _verify_inventory(robot: Robot, order: Order) -> None:
    missing = {s: q - robot.inventory.get(s, 0)
               for s, q in order.items.items()
               if robot.inventory.get(s, 0) < q}
    if missing:
        raise RuntimeError(
            f"Robot {robot.id} missing at fulfill: "
            + ", ".join(f"SKU {s} needs {q} more" for s, q in missing.items()))
    surplus = {s: q - order.items.get(s, 0)
               for s, q in robot.inventory.items()
               if q > order.items.get(s, 0)}
    if surplus:
        raise RuntimeError(
            f"Robot {robot.id} surplus inventory: "
            + ", ".join(f"SKU {s} +{q}" for s, q in surplus.items()))


def _check_stall(robot: Robot, tick: int, stall_limit: int, actions: List[ActionEntry]) -> None:
    recent_non_wait = sum(
        1 for a in reversed(actions)
        if a.robot_id == robot.id
        and a.tick >= tick - stall_limit
        and a.action in ("move", "pick")
    )
    if recent_non_wait == 0 and tick - robot.last_tick >= stall_limit:
        raise RuntimeError(
            f"Robot {robot.id} stalled {stall_limit}+ ticks at ({robot.x},{robot.y}) "
            f"tick={tick}")
