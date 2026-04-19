"""Cooperative nest construction scheduler.

Builds two rows of 20 pallets (10 at y=0, 10 at y=2) for each nest so that robots
can later cycle around them in the conveyor-belt order phase.

Pattern (nest_x = 15):
  y=0: (15,0) (16,0) ... (24,0)   <- row 0, placed first
  y=1: (conveyor lane, stays empty)
  y=2: (15,2) (16,2) ... (24,2)   <- row 2, placed second

Each robot is assigned to exactly one nest and never approaches another nest's zone.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from jr_walker.entities import ActionEntry, JobKind, Pallet, Robot
from jr_walker.pathfinder import (
    find_clearing,
    find_path,
    iter_clearings,
    manhattan,
    nearest_available_robot,
    nearest_free_adjacent,
    nearest_pallet_for_dest,
    reserve_path,
)
from jr_walker.warehouse import SpacetimeGrid


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def assign_robots_to_nests(robots: List[Robot], nest_x_coords: List[int]) -> None:
    """Round-robin: for each nest slot, assign the nearest unassigned robot.

    Modifies robot.nest_id in-place. Raises if any nest ends up with no robots.
    """
    unassigned = list(robots)
    nest_counts: Dict[int, int] = {nx: 0 for nx in nest_x_coords}

    while unassigned:
        for nest_x in nest_x_coords:
            if not unassigned:
                break
            anchor_x = nest_x + 5
            anchor_y = 3
            best = min(unassigned, key=lambda r: manhattan(r.x, r.y, anchor_x, anchor_y))
            best.nest_id = nest_x
            nest_counts[nest_x] += 1
            unassigned.remove(best)

    for nx, count in nest_counts.items():
        if count == 0:
            raise RuntimeError(f"Nest at x={nx} has no assigned robots — cannot build.")


def plan_nest_construction(
    nest_x: int,
    nest_robots: List[Robot],
    pallets: List[Pallet],
    grid: SpacetimeGrid,
    actions: List[ActionEntry],
    other_nest_rects: List[Tuple[int, int, int, int]],
    strict_no_swap: bool,
    total_robot_count: int,
) -> int:
    """Plan all pallet-placement jobs for one nest. Returns tick when last pallet lands."""
    planned_ids: Set[int] = set()
    unplanned_skus: Set[int] = set(range(1, 21))

    row0_coords = [(nest_x + i, 0) for i in range(10)]
    row2_coords = [(nest_x + i, 2) for i in range(10)]

    _plan_row(
        dest_coords=row0_coords,
        nest_robots=nest_robots,
        pallets=pallets,
        grid=grid,
        actions=actions,
        planned_ids=planned_ids,
        unplanned_skus=unplanned_skus,
        other_nest_rects=other_nest_rects,
        strict_no_swap=strict_no_swap,
        total_robot_count=total_robot_count,
        is_last_row=False,
    )

    last_tick = _plan_row(
        dest_coords=row2_coords,
        nest_robots=nest_robots,
        pallets=pallets,
        grid=grid,
        actions=actions,
        planned_ids=planned_ids,
        unplanned_skus=unplanned_skus,
        other_nest_rects=other_nest_rects,
        strict_no_swap=strict_no_swap,
        total_robot_count=total_robot_count,
        is_last_row=True,
    )
    return last_tick


def _plan_row(
    dest_coords: List[Tuple[int, int]],
    nest_robots: List[Robot],
    pallets: List[Pallet],
    grid: SpacetimeGrid,
    actions: List[ActionEntry],
    planned_ids: Set[int],
    unplanned_skus: Set[int],
    other_nest_rects: List[Tuple[int, int, int, int]],
    strict_no_swap: bool,
    total_robot_count: int,
    is_last_row: bool,
) -> int:
    placement_ticks: List[int] = []

    for dest_x, dest_y in dest_coords:
        pallet = nearest_pallet_for_dest(dest_x, dest_y, pallets, planned_ids, unplanned_skus)
        if pallet is None:
            raise RuntimeError(f"No pallet available for nest dest ({dest_x},{dest_y})")

        robot = nearest_available_robot(pallet.x, pallet.y, nest_robots)
        if robot is None:
            raise RuntimeError("No robots available for nest construction")

        finish_tick = _plan_setup_job(
            robot=robot,
            pallet=pallet,
            dest_x=dest_x,
            dest_y=dest_y,
            grid=grid,
            actions=actions,
            other_nest_rects=other_nest_rects,
            strict_no_swap=strict_no_swap,
        )

        planned_ids.add(pallet.id)
        unplanned_skus.discard(pallet.sku)
        placement_ticks.append(finish_tick)

    if is_last_row and placement_ticks:
        global_finish = max(placement_ticks)
        _emit_post_placement_waits(nest_robots, global_finish, actions, grid)
        return global_finish

    return max(placement_ticks, default=0)


def _emit_post_placement_waits(
    nest_robots: List[Robot],
    global_finish: int,
    actions: List[ActionEntry],
    grid: SpacetimeGrid,
) -> None:
    for robot in nest_robots:
        if robot.last_tick < global_finish:
            _emit_wait(robot, global_finish, grid, actions)


# ---------------------------------------------------------------------------
# Single setup-job planner
# ---------------------------------------------------------------------------

def _plan_setup_job(
    robot: Robot,
    pallet: Pallet,
    dest_x: int,
    dest_y: int,
    grid: SpacetimeGrid,
    actions: List[ActionEntry],
    other_nest_rects: List[Tuple[int, int, int, int]],
    strict_no_swap: bool,
) -> int:
    """Plan robot path to dock with pallet, push it to dest, and undock.

    Returns the tick at which the undock completes (pallet is in final position).
    """
    approach_cell = nearest_free_adjacent(pallet.x, pallet.y, robot.last_tick + 1, grid)
    if approach_cell is None:
        raise RuntimeError(f"No adjacent free cell for pallet {pallet.id} at ({pallet.x},{pallet.y})")

    _navigate_robot(robot, approach_cell[0], approach_cell[1], grid, actions,
                    other_nest_rects, strict_no_swap)

    # Dock with the pallet
    dock_tick = robot.last_tick + 1
    actions.append(ActionEntry(dock_tick, robot.id, "dock", pallet.x, pallet.y))
    grid.reserve(dock_tick, robot.x, robot.y)
    grid.clear_from(dock_tick + 1, pallet.x, pallet.y)
    robot.last_tick = dock_tick

    dock_dx = pallet.x - robot.x
    dock_dy = pallet.y - robot.y
    robot.docks[(dock_dx, dock_dy)] = pallet.id

    # Reorient so pallet is NORTH of robot (offset (0, -1))
    if (dock_dx, dock_dy) != (0, -1):
        _reorient_to_north(robot, pallet, grid, actions, other_nest_rects, strict_no_swap)

    # Push pallet north to destination: robot ends at (dest_x, dest_y + 1)
    _push_to_dest(robot, pallet, dest_x, dest_y, grid, actions, other_nest_rects, strict_no_swap)

    # Undock: pallet stays at (dest_x, dest_y) permanently
    undock_tick = robot.last_tick + 1
    actions.append(ActionEntry(undock_tick, robot.id, "undock", dest_x, dest_y))
    grid.reserve(undock_tick, robot.x, robot.y)
    grid.reserve_permanent(undock_tick, dest_x, dest_y)
    del robot.docks[(0, -1)]
    robot.last_tick = undock_tick

    return undock_tick


# ---------------------------------------------------------------------------
# Reorientation: get pallet to be NORTH of robot
# ---------------------------------------------------------------------------

def _reorient_to_north(
    robot: Robot,
    pallet: Pallet,
    grid: SpacetimeGrid,
    actions: List[ActionEntry],
    ex_rects: List[Tuple[int, int, int, int]],
    strict_no_swap: bool,
) -> None:
    """Move robot+pallet to clear space, undock, go south of pallet, redock."""
    offset = _current_dock_offset(robot, pallet.id)

    # Find the nearest clearing that the docked robot can actually reach.
    # The first candidate (find_clearing) may be unreachable if another robot's
    # spacetime reservation blocks every path to it — so we probe each candidate
    # in distance order and commit to the first one with a valid path.
    navigated = False
    for cx, cy in iter_clearings(pallet.x, pallet.y, robot.last_tick + 1, 15, grid, search_radius=12):
        crx, cry = cx - offset[0], cy - offset[1]
        if (crx, cry) == (robot.x, robot.y):
            navigated = True
            break
        path = find_path(robot.x, robot.y, robot.last_tick, crx, cry, grid,
                         footprint_offsets=[offset], strict_no_swap=strict_no_swap,
                         excluded_rects=ex_rects)
        if path:
            _apply_path(robot, path, [offset], grid, actions)
            pallet.x = robot.x + offset[0]
            pallet.y = robot.y + offset[1]
            navigated = True
            break

    if not navigated:
        # Hard fallback: fixed position above pallet, let _navigate_robot_docked raise if stuck.
        clearing = (pallet.x, max(1, pallet.y - 3))
        crx, cry = clearing[0] - offset[0], clearing[1] - offset[1]
        if (crx, cry) != (robot.x, robot.y):
            _navigate_robot_docked(robot, pallet, crx, cry, grid, actions, ex_rects, strict_no_swap)

    cur_offset = _current_dock_offset(robot, pallet.id)
    cur_pallet_x = robot.x + cur_offset[0]
    cur_pallet_y = robot.y + cur_offset[1]

    undock_tick = robot.last_tick + 1
    actions.append(ActionEntry(undock_tick, robot.id, "undock", cur_pallet_x, cur_pallet_y))
    grid.reserve(undock_tick, robot.x, robot.y)
    del robot.docks[cur_offset]
    grid.reserve_permanent(undock_tick, cur_pallet_x, cur_pallet_y)
    robot.last_tick = undock_tick

    south_x, south_y = cur_pallet_x, cur_pallet_y + 1
    _navigate_robot(robot, south_x, south_y, grid, actions, ex_rects, strict_no_swap,
                    extra_obstacles=frozenset([(cur_pallet_x, cur_pallet_y)]))

    redock_tick = robot.last_tick + 1
    grid.clear_from(redock_tick + 1, cur_pallet_x, cur_pallet_y)
    actions.append(ActionEntry(redock_tick, robot.id, "dock", cur_pallet_x, cur_pallet_y))
    grid.reserve(redock_tick, robot.x, robot.y)
    robot.docks[(0, -1)] = pallet.id
    pallet.x, pallet.y = cur_pallet_x, cur_pallet_y
    robot.last_tick = redock_tick


def _current_dock_offset(robot: Robot, pallet_id: int) -> Tuple[int, int]:
    for offset, pid in robot.docks.items():
        if pid == pallet_id:
            return offset
    raise RuntimeError(f"Pallet {pallet_id} not docked to robot {robot.id}")


# ---------------------------------------------------------------------------
# Movement helpers
# ---------------------------------------------------------------------------

def _navigate_robot(
    robot: Robot,
    goal_x: int,
    goal_y: int,
    grid: SpacetimeGrid,
    actions: List[ActionEntry],
    ex_rects: List[Tuple[int, int, int, int]],
    strict_no_swap: bool,
    extra_obstacles: FrozenSet[Tuple[int, int]] = frozenset(),
) -> None:
    if (robot.x, robot.y) == (goal_x, goal_y):
        return
    path = find_path(robot.x, robot.y, robot.last_tick, goal_x, goal_y, grid,
                     strict_no_swap=strict_no_swap, excluded_rects=ex_rects,
                     extra_obstacles=extra_obstacles)
    if not path:
        raise RuntimeError(
            f"Robot {robot.id} cannot reach ({goal_x},{goal_y}) from ({robot.x},{robot.y}) "
            f"at t={robot.last_tick}")
    _apply_path(robot, path, [], grid, actions)


def _navigate_robot_docked(
    robot: Robot,
    pallet: Pallet,
    goal_x: int,
    goal_y: int,
    grid: SpacetimeGrid,
    actions: List[ActionEntry],
    ex_rects: List[Tuple[int, int, int, int]],
    strict_no_swap: bool,
) -> None:
    if (robot.x, robot.y) == (goal_x, goal_y):
        return
    offset = _current_dock_offset(robot, pallet.id)
    path = find_path(robot.x, robot.y, robot.last_tick, goal_x, goal_y, grid,
                     footprint_offsets=[offset], strict_no_swap=strict_no_swap,
                     excluded_rects=ex_rects)
    if not path:
        raise RuntimeError(
            f"Robot {robot.id} (docked) cannot reach ({goal_x},{goal_y}) "
            f"from ({robot.x},{robot.y}) at t={robot.last_tick}")
    _apply_path(robot, path, [offset], grid, actions)
    pallet.x = robot.x + offset[0]
    pallet.y = robot.y + offset[1]


def _push_to_dest(
    robot: Robot,
    pallet: Pallet,
    dest_x: int,
    dest_y: int,
    grid: SpacetimeGrid,
    actions: List[ActionEntry],
    ex_rects: List[Tuple[int, int, int, int]],
    strict_no_swap: bool,
) -> None:
    _navigate_robot_docked(robot, pallet, dest_x, dest_y + 1,
                           grid, actions, ex_rects, strict_no_swap)


def _apply_path(
    robot: Robot,
    path: List[Tuple[int, int, int]],
    footprint_offsets: List[Tuple[int, int]],
    grid: SpacetimeGrid,
    actions: List[ActionEntry],
) -> None:
    for i, (t, x, y) in enumerate(path):
        if i == 0:
            continue
        actions.append(ActionEntry(t, robot.id, "move", x, y))
        grid.reserve(t, x, y)
        for dx, dy in footprint_offsets:
            grid.reserve(t, x + dx, y + dy)
    if path:
        _, last_x, last_y = path[-1]
        robot.x = last_x
        robot.y = last_y
        robot.last_tick = path[-1][0]


def _emit_wait(
    robot: Robot,
    until_tick: int,
    grid: SpacetimeGrid,
    actions: List[ActionEntry],
) -> None:
    for t in range(robot.last_tick + 1, until_tick + 1):
        if grid.is_free(t, robot.x, robot.y):
            grid.reserve(t, robot.x, robot.y)
    robot.last_tick = until_tick
