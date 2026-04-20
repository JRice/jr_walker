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

from jr_walker.entities import ActionEntry, JobKind, NestConfig, Pallet, Robot
from jr_walker.pathfinder import (
    find_clearing,
    find_path,
    iter_clearings,
    manhattan,
    nearest_available_robot,
    nearest_free_adjacent,
    reserve_path,
)
from jr_walker.warehouse import SpacetimeGrid


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def assign_robots_to_nests(
    robots: List[Robot],
    nest_configs: List[NestConfig],
) -> None:
    """Round-robin: for each nest slot, assign the nearest unassigned robot."""
    unassigned = list(robots)
    nest_counts: Dict[Tuple[int, int], int] = {nc.anchor: 0 for nc in nest_configs}

    while unassigned:
        for nc in nest_configs:
            if not unassigned:
                break
            nx, ny = nc.anchor
            anchor_x = nx + nc.n_positions // 2
            anchor_y = ny + 3
            best = min(unassigned, key=lambda r: manhattan(r.x, r.y, anchor_x, anchor_y))
            best.nest_id = nc.anchor
            nest_counts[nc.anchor] += 1
            unassigned.remove(best)

    for anchor, count in nest_counts.items():
        if count == 0:
            raise RuntimeError(
                f"Nest at {anchor} has no assigned robots — cannot build."
            )


def plan_nest_construction(
    nest_config: NestConfig,
    nest_robots: List[Robot],
    pallets: List[Pallet],
    grid: SpacetimeGrid,
    actions: List[ActionEntry],
    other_nest_rects: List[Tuple[int, int, int, int]],
    strict_no_swap: bool,
    total_robot_count: int,
    planned_ids: Optional[Set[int]] = None,
) -> int:
    """Plan all pallet-placement jobs for one nest. Returns tick when last pallet lands."""
    nx, ny = nest_config.anchor
    n = nest_config.n_positions

    # Line A: non-gap positions only (gaps mirror Line C gap positions)
    line_a_items = [
        (nx + i, ny, sku)
        for i, sku in enumerate(nest_config.line_a_pallets)
        if sku != 0
    ]
    line_a_coords = [(x, y) for x, y, _ in line_a_items]
    line_a_dest_skus = [sku for _, _, sku in line_a_items]

    # Line C: may have gaps — placed second
    line_c_items = [
        (nx + i, ny + 2, sku)
        for i, sku in enumerate(nest_config.line_c_pallets)
        if sku != 0
    ]
    line_c_coords = [(x, y) for x, y, _ in line_c_items]
    line_c_dest_skus = [sku for _, _, sku in line_c_items]

    if planned_ids is None:
        planned_ids = set()

    _plan_row(
        dest_coords=line_a_coords,
        dest_skus=line_a_dest_skus,
        nest_robots=nest_robots,
        pallets=pallets,
        grid=grid,
        actions=actions,
        planned_ids=planned_ids,
        other_nest_rects=other_nest_rects,
        strict_no_swap=strict_no_swap,
        total_robot_count=total_robot_count,
        is_last_row=False,
    )

    last_tick = _plan_row(
        dest_coords=line_c_coords,
        dest_skus=line_c_dest_skus,
        nest_robots=nest_robots,
        pallets=pallets,
        grid=grid,
        actions=actions,
        planned_ids=planned_ids,
        other_nest_rects=other_nest_rects,
        strict_no_swap=strict_no_swap,
        total_robot_count=total_robot_count,
        is_last_row=True,
    )
    return last_tick


def _nearest_pallet_with_sku(
    dest_x: int,
    dest_y: int,
    pallets: List[Pallet],
    planned_ids: Set[int],
    target_sku: int,
) -> Optional[Pallet]:
    """Nearest unplanned pallet that carries exactly target_sku."""
    best: Optional[Pallet] = None
    best_dist = 10**9
    for p in pallets:
        if p.id in planned_ids or p.sku != target_sku:
            continue
        d = manhattan(p.x, p.y, dest_x, dest_y)
        if d < best_dist:
            best_dist = d
            best = p
    return best


def _plan_row(
    dest_coords: List[Tuple[int, int]],
    dest_skus: List[int],
    nest_robots: List[Robot],
    pallets: List[Pallet],
    grid: SpacetimeGrid,
    actions: List[ActionEntry],
    planned_ids: Set[int],
    other_nest_rects: List[Tuple[int, int, int, int]],
    strict_no_swap: bool,
    total_robot_count: int,
    is_last_row: bool,
) -> int:
    placement_ticks: List[int] = []

    # Hold all robots at their current positions so that robots planned later in
    # this row route around them rather than passing through.
    for r in nest_robots:
        grid.hold_for(r.last_tick + 1, r.x, r.y)

    for (dest_x, dest_y), target_sku in zip(dest_coords, dest_skus):
        pallet = _nearest_pallet_with_sku(dest_x, dest_y, pallets, planned_ids, target_sku)
        if pallet is None:
            raise RuntimeError(
                f"No pallet with SKU {target_sku} available for nest dest ({dest_x},{dest_y})"
            )

        robot = nearest_available_robot(pallet.x, pallet.y, nest_robots)
        if robot is None:
            raise RuntimeError("No robots available for nest construction")

        # Release the active robot's hold — it is about to plan its own path.
        grid.release_hold(robot.last_tick + 1, robot.x, robot.y)

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

        # Re-hold the robot at its new resting position for subsequent iterations.
        grid.hold_for(robot.last_tick + 1, robot.x, robot.y)

        planned_ids.add(pallet.id)
        # no unplanned_skus to update — each position has its own required SKU
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
    approach_cell = _pick_approach_cell(pallet, robot, grid)
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

    # Hold robot + pallet so subsequent robots don't plan through this position.
    grid.hold_for(dock_tick + 1, robot.x, robot.y)
    grid.hold_for(dock_tick + 1, pallet.x, pallet.y)

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

    # Hold robot's resting position until it is assigned another job.
    grid.hold_for(undock_tick + 1, robot.x, robot.y)

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
    # Release the robot + pallet holds placed after docking so they don't block
    # the path search (the robot is about to move, so the holds are stale).
    grid.release_hold(robot.last_tick + 1, robot.x, robot.y)
    grid.release_hold(robot.last_tick + 1, pallet.x, pallet.y)

    navigated = False
    for cx, cy in iter_clearings(pallet.x, pallet.y, robot.last_tick + 1, 15, grid, search_radius=12):
        # Reorientation requires the robot to reach the cell south of the clearing
        # center (cx, cy+1) after the intermediate undock.  Skip any clearing
        # whose south cell is permanently inaccessible (e.g. an original pallet
        # that was never picked up).
        south_cand_y = cy + 1
        t_chk = robot.last_tick + 2
        if (not grid.in_bounds(cx, south_cand_y)
                or (t_chk < grid.max_ticks
                    and bool(grid.grid[t_chk:, south_cand_y, cx].all()))):
            continue

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
        # Fallback: try positions progressively further north of the pallet,
        # verifying south accessibility before committing.
        for dy_try in range(3, 10):
            fx, fy = pallet.x, max(1, pallet.y - dy_try)
            frx, fry = fx - offset[0], fy - offset[1]
            south_fb_y = fy + 1
            if not grid.in_bounds(fx, south_fb_y) or not grid.in_bounds(frx, fry):
                continue
            t_chk = robot.last_tick + 2
            if t_chk < grid.max_ticks and bool(grid.grid[t_chk:, south_fb_y, fx].all()):
                continue  # south permanently blocked
            if (frx, fry) == (robot.x, robot.y):
                navigated = True
                break
            path = find_path(robot.x, robot.y, robot.last_tick, frx, fry, grid,
                             footprint_offsets=[offset], strict_no_swap=strict_no_swap,
                             excluded_rects=ex_rects)
            if path:
                _apply_path(robot, path, [offset], grid, actions)
                pallet.x = robot.x + offset[0]
                pallet.y = robot.y + offset[1]
                navigated = True
                break

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
    grid.release_hold(robot.last_tick + 1, robot.x, robot.y)
    path = find_path(robot.x, robot.y, robot.last_tick, goal_x, goal_y, grid,
                     strict_no_swap=strict_no_swap, excluded_rects=ex_rects,
                     extra_obstacles=extra_obstacles)
    if not path:
        raise RuntimeError(
            f"Robot {robot.id} cannot reach ({goal_x},{goal_y}) from ({robot.x},{robot.y}) "
            f"at t={robot.last_tick}")
    _apply_path(robot, path, [], grid, actions)
    grid.hold_for(robot.last_tick + 1, robot.x, robot.y)


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
    grid.release_hold(robot.last_tick + 1, robot.x, robot.y)
    grid.release_hold(robot.last_tick + 1, pallet.x, pallet.y)
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
    grid.hold_for(robot.last_tick + 1, robot.x, robot.y)
    grid.hold_for(robot.last_tick + 1, pallet.x, pallet.y)


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
        _, px, py = path[i - 1]
        if x != px or y != py:  # skip wait-in-place steps (same cell = no-op move)
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


def _pick_approach_cell(
    pallet: Pallet,
    robot: Robot,
    grid: SpacetimeGrid,
    headroom: int = 10,
) -> Optional[Tuple[int, int]]:
    """Choose an adjacent cell to the pallet that will still be free for `headroom`
    ticks after the robot's estimated arrival.  Picks the first candidate with
    temporal headroom; falls back to any cell free at robot.last_tick + 1.

    Estimating arrival time as manhattan distance avoids choosing a cell that
    another robot will shortly occupy, which would leave our robot trapped after
    docking with no space to manoeuvre.
    """
    # south preferred first (natural push-north direction), then west, east, north
    candidates = [
        (pallet.x, pallet.y + 1),
        (pallet.x - 1, pallet.y),
        (pallet.x + 1, pallet.y),
        (pallet.x, pallet.y - 1),
    ]
    t_now = robot.last_tick + 1
    # Pass 1: cell free for [est_arrival .. est_arrival + headroom)
    for cx, cy in candidates:
        if not grid.in_bounds(cx, cy):
            continue
        est = t_now + manhattan(robot.x, robot.y, cx, cy)
        if all(grid.is_free(est + dt, cx, cy) for dt in range(headroom)):
            return cx, cy
    # Pass 2: any cell free right now (original behaviour)
    for cx, cy in candidates:
        if grid.in_bounds(cx, cy) and grid.is_free(t_now, cx, cy):
            return cx, cy
    return None
