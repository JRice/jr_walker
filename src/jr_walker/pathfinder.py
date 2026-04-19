from __future__ import annotations

import heapq
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from jr_walker.warehouse import SpacetimeGrid


# Cardinal moves + wait-in-place; order matters for tie-breaking (prefer movement)
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]


def manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def adjacent_cells(x: int, y: int, width: int, height: int) -> List[Tuple[int, int]]:
    return [
        (x + dx, y + dy)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if 0 <= x + dx < width and 0 <= y + dy < height
    ]


# ---------------------------------------------------------------------------
# Spacetime A* (greedy single-robot)
# ---------------------------------------------------------------------------

def find_path(
    start_x: int,
    start_y: int,
    start_t: int,
    goal_x: int,
    goal_y: int,
    grid: SpacetimeGrid,
    footprint_offsets: List[Tuple[int, int]] = [],
    strict_no_swap: bool = False,
    excluded_rects: List[Tuple[int, int, int, int]] = [],
    extra_obstacles: FrozenSet[Tuple[int, int]] = frozenset(),
) -> List[Tuple[int, int, int]]:
    """Spacetime A* — returns [(t, x, y), ...] from start to goal, or [] if unreachable.

    footprint_offsets: additional cells (dx, dy) that must be free at each step
                       (used for dock-aware pathfinding with a pallet attached north).
    strict_no_swap:    also require the target cell to be free at time t (not just t+1),
                       preventing two entities from passing through each other.
    excluded_rects:    rectangles (x1, y1, x2, y2) inclusive that are always blocked
                       (used to keep robots away from other nests).
    extra_obstacles:   additional (x, y) cells treated as permanently blocked.
    """

    def cell_navigable(t: int, x: int, y: int) -> bool:
        if (x, y) in extra_obstacles:
            return False
        for x1, y1, x2, y2 in excluded_rects:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return False
        return grid.is_free(t, x, y)

    def state_navigable(t: int, x: int, y: int) -> bool:
        if not cell_navigable(t, x, y):
            return False
        for dx, dy in footprint_offsets:
            if not cell_navigable(t, x + dx, y + dy):
                return False
        return True

    h0 = manhattan(start_x, start_y, goal_x, goal_y)
    # heap: (f, g, t, x, y)
    heap: List[Tuple[int, int, int, int, int]] = [(h0, 0, start_t, start_x, start_y)]
    came_from: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {
        (start_t, start_x, start_y): None
    }
    g_score: Dict[Tuple[int, int, int], int] = {(start_t, start_x, start_y): 0}

    while heap:
        _, g, t, x, y = heapq.heappop(heap)

        if x == goal_x and y == goal_y:
            return _reconstruct(came_from, t, x, y)

        if g > g_score.get((t, x, y), 10**9):
            continue  # stale entry

        nt = t + 1
        if nt >= grid.max_ticks:
            continue

        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if not grid.in_bounds(nx, ny):
                continue
            if not state_navigable(nt, nx, ny):
                continue
            # Strict swap prevention: target must also be free at current tick
            if strict_no_swap and (dx, dy) != (0, 0):
                if not grid.is_free(t, nx, ny):
                    continue

            state = (nt, nx, ny)
            ng = g + 1
            if ng < g_score.get(state, 10**9):
                g_score[state] = ng
                came_from[state] = (t, x, y)
                h = manhattan(nx, ny, goal_x, goal_y)
                heapq.heappush(heap, (ng + h, ng, nt, nx, ny))

    return []


def _reconstruct(
    came_from: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]],
    t: int,
    x: int,
    y: int,
) -> List[Tuple[int, int, int]]:
    path: List[Tuple[int, int, int]] = []
    state: Optional[Tuple[int, int, int]] = (t, x, y)
    while state is not None:
        path.append(state)
        state = came_from[state]
    path.reverse()
    return path


def reserve_path(
    path: List[Tuple[int, int, int]],
    grid: SpacetimeGrid,
    footprint_offsets: List[Tuple[int, int]] = [],
) -> None:
    """Burn a planned path into the reservation grid (robot body + any docked pallets)."""
    for t, x, y in path:
        grid.reserve(t, x, y)
        for dx, dy in footprint_offsets:
            grid.reserve(t, x + dx, y + dy)


# ---------------------------------------------------------------------------
# Nearest-pallet helpers
# ---------------------------------------------------------------------------

def nearest_pallet_for_dest(
    dest_x: int,
    dest_y: int,
    pallets: List,
    planned_ids: Set[int],
    unplanned_skus: Set[int],
) -> Optional[object]:
    """Nearest pallet (by manhattan to dest) that is unplanned and has a needed SKU."""
    best = None
    best_dist = 10**9
    for p in pallets:
        if p.id in planned_ids or p.sku not in unplanned_skus:
            continue
        d = manhattan(p.x, p.y, dest_x, dest_y)
        if d < best_dist:
            best_dist = d
            best = p
    return best


def nearest_available_robot(
    target_x: int,
    target_y: int,
    robots: List,
) -> Optional[object]:
    """Robot with minimum (last_tick + 1 + manhattan_to_target) — the soonest it can arrive."""
    best = None
    best_score = 10**9
    for r in robots:
        score = (r.last_tick + 1) + manhattan(r.x, r.y, target_x, target_y)
        if score < best_score:
            best_score = score
            best = r
    return best


def nearest_free_adjacent(
    pallet_x: int,
    pallet_y: int,
    t: int,
    grid: SpacetimeGrid,
    prefer_south: bool = True,
) -> Optional[Tuple[int, int]]:
    """Adjacent cell to the pallet that is free at tick t. South is tried first when preferred."""
    order = (
        [(pallet_x, pallet_y + 1), (pallet_x - 1, pallet_y),
         (pallet_x + 1, pallet_y), (pallet_x, pallet_y - 1)]
        if prefer_south
        else [(pallet_x, pallet_y - 1), (pallet_x - 1, pallet_y),
              (pallet_x + 1, pallet_y), (pallet_x, pallet_y + 1)]
    )
    for cx, cy in order:
        if grid.in_bounds(cx, cy) and grid.is_free(t, cx, cy):
            return cx, cy
    return None


def find_clearing(
    center_x: int,
    center_y: int,
    t_start: int,
    ticks_needed: int,
    grid: SpacetimeGrid,
    search_radius: int = 12,
) -> Optional[Tuple[int, int]]:
    """Nearest (x, y) where the 3x3 patch is clear for ticks_needed ticks from t_start.

    Searches outward by manhattan distance from center.
    """
    return next(iter_clearings(center_x, center_y, t_start, ticks_needed, grid, search_radius), None)


def iter_clearings(
    center_x: int,
    center_y: int,
    t_start: int,
    ticks_needed: int,
    grid: SpacetimeGrid,
    search_radius: int = 12,
):
    """Yield (x, y) positions where the 3x3 patch is clear, in ascending manhattan order."""
    for r in range(search_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if abs(dx) + abs(dy) != r:
                    continue
                cx, cy = center_x + dx, center_y + dy
                if grid.is_3x3_clear(cx, cy, t_start, ticks_needed):
                    yield cx, cy
