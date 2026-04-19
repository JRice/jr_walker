"""Tests for A* pathfinder."""
import pytest
from jr_walker.pathfinder import find_path, manhattan, nearest_pallet_for_dest, reserve_path
from jr_walker.warehouse import SpacetimeGrid


def empty_grid(w=20, h=20, t=200):
    return SpacetimeGrid(w, h, t)


# ---------------------------------------------------------------------------
# manhattan
# ---------------------------------------------------------------------------

def test_manhattan_same_point():
    assert manhattan(3, 4, 3, 4) == 0


def test_manhattan_basic():
    assert manhattan(0, 0, 3, 4) == 7
    assert manhattan(5, 5, 2, 1) == 7


# ---------------------------------------------------------------------------
# find_path
# ---------------------------------------------------------------------------

def test_trivial_path_already_at_goal():
    g = empty_grid()
    path = find_path(5, 5, 0, 5, 5, g)
    assert path == [(0, 5, 5)]


def test_straight_path_east():
    g = empty_grid()
    path = find_path(0, 5, 0, 3, 5, g)
    assert path is not None
    assert len(path) == 4  # start + 3 moves
    assert path[0] == (0, 0, 5)
    assert path[-1] == (3, 3, 5)


def test_path_around_obstacle():
    g = empty_grid()
    # Block direct east path at x=1, y=5
    for t in range(200):
        g.reserve(t, 1, 5)
    path = find_path(0, 5, 0, 2, 5, g)
    assert path is not None
    # Path should go around, not through (1,5)
    positions = [(x, y) for _, x, y in path]
    assert (1, 5) not in positions
    assert path[-1][1:] == (2, 5)


def test_no_path_fully_blocked():
    g = empty_grid(w=5, h=5)
    # Surround (2,2) completely at all times
    for t in range(200):
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            g.reserve(t, 2 + dx, 2 + dy)
    g.reserve_permanent(0, 2, 2)
    path = find_path(0, 0, 0, 2, 2, g)
    assert path == []


def test_path_with_footprint_offset():
    g = empty_grid()
    # Robot at (0,5) with pallet at (0,4) [north, offset (0,-1)]
    # Navigate to (3,5) — pallet must also be free at each tick
    # Block (2,4) at all times so path must go around
    for t in range(200):
        g.reserve(t, 2, 4)
    path = find_path(0, 5, 0, 3, 5, g, footprint_offsets=[(0, -1)])
    assert path is not None
    # Path should avoid (2,4) for the pallet
    for t, x, y in path:
        assert (x, y) != (2, 4)         # robot body
        assert (x, y - 1) != (2, 4)     # pallet


def test_strict_no_swap_blocks_swap():
    g = empty_grid()
    # Reserve (2,5) at t=0 and (1,5) at t=1 — simulates entity moving west
    g.reserve(0, 2, 5)
    g.reserve(1, 1, 5)
    # With strict mode: robot at (1,5) trying to move east at t=1 to (2,5)
    # (2,5) was occupied at t=0, so strict mode blocks it
    path = find_path(1, 5, 0, 3, 5, g, strict_no_swap=True)
    # Should still find path eventually (wait then move)
    assert path is not None
    assert path[-1][1:] == (3, 5)


def test_excluded_rect_blocked():
    g = empty_grid()
    # Goal is inside excluded rect — should fail
    path = find_path(0, 0, 0, 5, 5, g, excluded_rects=[(3, 3, 7, 7)])
    assert path == []


def test_reserve_path_marks_grid():
    g = empty_grid()
    path = [(0, 0, 0), (1, 1, 0), (2, 2, 0)]
    reserve_path(path, g)
    assert not g.is_free(0, 0, 0)
    assert not g.is_free(1, 1, 0)
    assert not g.is_free(2, 2, 0)
    assert g.is_free(3, 3, 0)  # after path


# ---------------------------------------------------------------------------
# nearest_pallet_for_dest
# ---------------------------------------------------------------------------

def test_nearest_pallet_for_dest_basic():
    from jr_walker.entities import Pallet
    pallets = [
        Pallet(id=0, x=10, y=10, sku=1),
        Pallet(id=1, x=2, y=2, sku=2),
        Pallet(id=2, x=5, y=5, sku=1),
    ]
    result = nearest_pallet_for_dest(0, 0, pallets, planned_ids=set(), unplanned_skus={1, 2})
    assert result is not None
    assert result.id == 1  # (2,2) is closest to (0,0)


def test_nearest_pallet_skips_planned():
    from jr_walker.entities import Pallet
    pallets = [Pallet(id=0, x=1, y=1, sku=5), Pallet(id=1, x=5, y=5, sku=5)]
    result = nearest_pallet_for_dest(0, 0, pallets, planned_ids={0}, unplanned_skus={5})
    assert result is not None
    assert result.id == 1  # id=0 is planned


def test_nearest_pallet_skips_wrong_sku():
    from jr_walker.entities import Pallet
    pallets = [Pallet(id=0, x=1, y=1, sku=7), Pallet(id=1, x=3, y=3, sku=5)]
    result = nearest_pallet_for_dest(0, 0, pallets, planned_ids=set(), unplanned_skus={5})
    assert result is not None
    assert result.id == 1
