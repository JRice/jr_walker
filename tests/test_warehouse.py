"""Tests for SpacetimeGrid reservation logic."""
import pytest
from jr_walker.warehouse import SpacetimeGrid


def make_grid(w=10, h=10, t=50):
    return SpacetimeGrid(w, h, t)


def test_initially_all_free():
    g = make_grid()
    assert g.is_free(0, 5, 5)
    assert g.is_free(49, 9, 9)


def test_reserve_marks_occupied():
    g = make_grid()
    g.reserve(3, 2, 4)
    assert not g.is_free(3, 2, 4)
    assert g.is_free(2, 2, 4)   # different tick
    assert g.is_free(3, 3, 4)   # different cell


def test_reserve_permanent_marks_all_future_ticks():
    g = make_grid()
    g.reserve_permanent(5, 3, 3)
    assert g.is_free(4, 3, 3)
    assert not g.is_free(5, 3, 3)
    assert not g.is_free(49, 3, 3)


def test_clear_from_removes_occupancy():
    g = make_grid()
    g.reserve_permanent(0, 1, 1)
    g.clear_from(3, 1, 1)
    assert not g.is_free(2, 1, 1)
    assert g.is_free(3, 1, 1)
    assert g.is_free(49, 1, 1)


def test_out_of_bounds_always_not_free():
    g = make_grid()
    assert not g.is_free(0, -1, 0)
    assert not g.is_free(0, 10, 0)
    assert not g.is_free(0, 0, 10)
    assert not g.is_free(50, 0, 0)  # t out of range


def test_in_bounds():
    g = make_grid(w=60, h=40)
    assert g.in_bounds(0, 0)
    assert g.in_bounds(59, 39)
    assert not g.in_bounds(60, 0)
    assert not g.in_bounds(0, 40)


def test_is_3x3_clear():
    g = make_grid(w=20, h=20, t=50)
    # Empty grid — any center should be clear
    assert g.is_3x3_clear(5, 5, 0, 5)

    # Block one cell in the 3x3 patch
    g.reserve(2, 6, 5)
    assert not g.is_3x3_clear(5, 5, 0, 5)

    # But at a different time range it's fine
    assert g.is_3x3_clear(5, 5, 3, 5)


def test_footprint_free():
    g = make_grid()
    positions = [(1, 1), (2, 1), (1, 2)]
    assert g.is_footprint_free(0, positions)
    g.reserve(0, 2, 1)
    assert not g.is_footprint_free(0, positions)
