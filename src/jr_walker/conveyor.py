"""Conveyor-belt order fulfillment phase.

Each robot orbits its assigned nest counter-clockwise on a fixed track, picking
exactly the SKUs it needs for the current order as it passes each nest pallet,
then fulfilling at the north perimeter before looping back.

Track for nest_x (robot starts and ends at (nest_x, 3)):

  Segment A — east  11 steps  (nest_x,3) -> (nest_x+11,3)
                               picks from y=2 pallets (x = nest_x..nest_x+9)
  Segment B — north  2 steps  -> (nest_x+11,1)
  Segment C — west  12 steps  -> (nest_x-1,1)
                               picks from y=0 pallets (x = nest_x..nest_x+9)
  Fulfill              step   north 1 -> (nest_x-1,0); fulfill(); west 1; south 3; east 2
                               -> back at (nest_x,3)

The robot stops and picks as many times as needed at each pallet cell before moving.
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
    """Ordered (x, y) positions of the full conveyor cycle."""
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

    # Line B (y=ny+1): pick from Line A (y=ny)
    if track_y == ny + 1 and 0 <= pos_idx < n:
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
    track = build_conveyor_track(nest_config)

    # Build SKU→position map from config (source of truth for what was placed where)
    sku_to_pos: Dict[int, Tuple[int, int]] = {}
    for i, sku in enumerate(nest_config.line_a_skus):
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

    # Spread robots evenly around the circular track so they don't all converge
    # on (nest_x, 3) at once and collide.  Robot[i] starts at track[i * spread].
    num_robots = len(sorted_robots)
    track_size = len(track)
    spread = track_size // max(num_robots, 1)

    # Hold all robots while we plan their individual navigation paths.
    for r in sorted_robots:
        grid.hold_for(r.last_tick + 1, r.x, r.y)

    robot_start_idx: Dict[int, int] = {}
    for i, robot in enumerate(sorted_robots):
        start_idx = (i * spread) % track_size
        robot_start_idx[robot.id] = start_idx
        tx, ty = track[start_idx]
        grid.release_hold(robot.last_tick + 1, robot.x, robot.y)
        if (robot.x, robot.y) != (tx, ty):
            _navigate_to_position(robot, tx, ty, grid, actions, other_nest_rects, strict_no_swap)
        grid.hold_for(robot.last_tick + 1, robot.x, robot.y)

    # Holds are only needed during navigation planning; release them all now.
    for r in sorted_robots:
        grid.release_hold(r.last_tick + 1, r.x, r.y)

    order_idx = 0
    robot_order_map: Dict[int, Optional[Order]] = {r.id: None for r in sorted_robots}
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
        # All orders fulfilled once every robot has no remaining work
        if all(robot_order_map.get(r.id) is None for r in sorted_robots):
            for r in sorted_robots:
                r.job = JobKind.DONE
            break

        tick += 1
        if tick >= max_ticks:
            raise RuntimeError(f"Exceeded max_ticks={max_ticks} during order phase")
        if time.time() - start_time > max_runtime_seconds:
            raise RuntimeError("Exceeded max_runtime during order phase")

        # Include ALL robots (active and done) so done robots stay visible to
        # collision checks and never become phantom obstacles on the track.
        robot_positions: Dict[int, Tuple[int, int]] = {r.id: (r.x, r.y) for r in sorted_robots}

        for robot in sorted_robots:
            if robot.last_tick >= tick:
                continue

            order = robot_order_map.get(robot.id)
            track_pos = robot_track_idx[robot.id]
            cur_x, cur_y = track[track_pos % len(track)]
            next_idx = (track_pos + 1) % len(track)
            next_x, next_y = track[next_idx]

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

            # Fulfill at (nest_x-1, 0) — only if inventory is complete.
            # If incomplete (robot started mid-track and hasn't orbited all pallets yet),
            # skip and continue orbiting to pick the remaining items.
            inventory_complete = not any(
                order.items.get(s, 0) > robot.inventory.get(s, 0)
                for s in order.items
            )
            if (cur_x, cur_y) == nest_config.fulfill_near and inventory_complete:
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
                else:
                    robot_order_map[robot.id] = None

                if (len(fulfilled) - last_progress_orders >= progress_order_interval
                        or tick - last_progress_tick >= progress_tick_interval):
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}] tick={tick}  fulfilled={len(fulfilled)}")
                    last_progress_orders = len(fulfilled)
                    last_progress_tick = tick
                continue

            # Attempt to move to next track position
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
