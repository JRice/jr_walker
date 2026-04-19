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

from jr_walker.entities import ActionEntry, JobKind, Order, Robot
from jr_walker.pathfinder import find_path, manhattan
from jr_walker.warehouse import SpacetimeGrid


# ---------------------------------------------------------------------------
# Track builder
# ---------------------------------------------------------------------------

def build_conveyor_track(nest_x: int) -> List[Tuple[int, int]]:
    """Ordered (x, y) positions of the full conveyor cycle."""
    track: List[Tuple[int, int]] = []

    # Segment A: east from nest_x to nest_x+11 (12 positions, 11 moves)
    for x in range(nest_x, nest_x + 12):
        track.append((x, 3))

    # Segment B: north from y=3 to y=1 (2 moves)
    track.append((nest_x + 11, 2))
    track.append((nest_x + 11, 1))

    # Segment C: west from nest_x+10 to nest_x-1 (12 positions, 12 moves)
    for x in range(nest_x + 10, nest_x - 2, -1):
        track.append((x, 1))

    # Return path: north, fulfill, west, south x3, east x2
    track.append((nest_x - 1, 0))   # fulfill here (perimeter y=0)
    track.append((nest_x - 2, 0))
    track.append((nest_x - 2, 1))
    track.append((nest_x - 2, 2))
    track.append((nest_x - 2, 3))
    track.append((nest_x - 1, 3))   # one step east before looping back to (nest_x,3)

    return track


def pallet_to_pick_from(track_x: int, track_y: int, nest_x: int) -> Optional[Tuple[int, int]]:
    """Return the nest pallet position adjacent-north of this track cell, if any."""
    if track_y == 3 and nest_x <= track_x <= nest_x + 9:
        return track_x, 2
    if track_y == 1 and nest_x <= track_x <= nest_x + 9:
        return track_x, 0
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plan_order_phase(
    nest_x: int,
    nest_robots: List[Robot],
    orders: List[Order],
    nest_pallets: List,
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
    track = build_conveyor_track(nest_x)
    sku_to_pos: Dict[int, Tuple[int, int]] = {p.sku: (p.x, p.y) for p in nest_pallets}

    pending_orders = sorted(
        [o for o in orders if not o.is_fulfilled],
        key=lambda o: o.total_picks,
    )
    fulfilled: List[Order] = []

    sorted_robots = sorted(nest_robots, key=lambda r: manhattan(r.x, r.y, nest_x, 3))

    for robot in sorted_robots:
        if (robot.x, robot.y) != (nest_x, 3):
            _navigate_to_entry(robot, nest_x, grid, actions, other_nest_rects, strict_no_swap)

    order_idx = 0
    robot_order_map: Dict[int, Optional[Order]] = {r.id: None for r in sorted_robots}
    robot_track_idx: Dict[int, int] = {r.id: 0 for r in sorted_robots}

    for robot in sorted_robots:
        if order_idx < len(pending_orders):
            o = pending_orders[order_idx]
            o.assigned_tick = robot.last_tick
            robot_order_map[robot.id] = o
            order_idx += 1

    tick = max(r.last_tick for r in sorted_robots)
    active_robots = list(sorted_robots)
    last_progress_tick = 0
    last_progress_orders = 0

    while active_robots:
        tick += 1
        if tick >= max_ticks:
            raise RuntimeError(f"Exceeded max_ticks={max_ticks} during order phase")
        if time.time() - start_time > max_runtime_seconds:
            raise RuntimeError("Exceeded max_runtime during order phase")

        robot_positions: Dict[int, Tuple[int, int]] = {r.id: (r.x, r.y) for r in active_robots}

        for robot in list(active_robots):
            order = robot_order_map.get(robot.id)
            if order is None:
                active_robots.remove(robot)
                robot.job = JobKind.DONE
                continue

            if robot.last_tick >= tick:
                continue

            track_pos = robot_track_idx[robot.id]
            cur_x, cur_y = track[track_pos % len(track)]
            next_idx = (track_pos + 1) % len(track)
            next_x, next_y = track[next_idx]

            # Pick from adjacent pallet if needed
            pallet_pos = pallet_to_pick_from(cur_x, cur_y, nest_x)
            if pallet_pos is not None and order is not None:
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

            # Fulfill at (nest_x-1, 0)
            if (cur_x, cur_y) == (nest_x - 1, 0):
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
                r2.id != robot.id and robot_positions.get(r2.id) == (next_x, next_y)
                for r2 in active_robots
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

def _navigate_to_entry(
    robot: Robot,
    nest_x: int,
    grid: SpacetimeGrid,
    actions: List[ActionEntry],
    ex_rects: List[Tuple[int, int, int, int]],
    strict_no_swap: bool,
) -> None:
    path = find_path(robot.x, robot.y, robot.last_tick, nest_x, 3, grid,
                     excluded_rects=ex_rects, strict_no_swap=strict_no_swap)
    if not path:
        raise RuntimeError(
            f"Robot {robot.id} cannot reach conveyor entry ({nest_x},3) "
            f"from ({robot.x},{robot.y}) at t={robot.last_tick}")
    for i, (t, x, y) in enumerate(path):
        if i == 0:
            continue
        actions.append(ActionEntry(t, robot.id, "move", x, y))
        grid.reserve(t, x, y)
    if path:
        _, lx, ly = path[-1]
        robot.x, robot.y = lx, ly
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
