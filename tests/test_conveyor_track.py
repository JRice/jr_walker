"""Tests for conveyor track geometry."""
import pytest
from jr_walker.conveyor import build_conveyor_track, pallet_to_pick_from


def test_track_starts_and_ends_correctly():
    track = build_conveyor_track(15)
    assert track[0] == (15, 3)           # start
    assert track[-1] == (14, 3)          # last position before wrapping back to (15,3)


def test_track_has_expected_length():
    track = build_conveyor_track(15)
    # 12 east + 2 north + 12 west + 1 north + 1 west + 3 south + 1 east = 32
    assert len(track) == 32


def test_track_positions_in_segment_a():
    track = build_conveyor_track(15)
    # First 12 positions should be eastward at y=3
    for i in range(12):
        x, y = track[i]
        assert y == 3
        assert x == 15 + i


def test_track_fulfillment_point():
    track = build_conveyor_track(15)
    # Fulfill at (14, 0)
    assert (14, 0) in track


def test_fulfill_point_is_perimeter():
    # y=0 is the perimeter for the 60x40 grid
    track = build_conveyor_track(15)
    fulfill_pos = next(pos for pos in track if pos[1] == 0 and pos[0] == 14)
    assert fulfill_pos == (14, 0)


def test_pallet_to_pick_from_segment_a():
    # Robot at (15..24, 3) should pick from (x, 2)
    assert pallet_to_pick_from(15, 3, 15) == (15, 2)
    assert pallet_to_pick_from(24, 3, 15) == (24, 2)
    assert pallet_to_pick_from(25, 3, 15) is None   # x=25 is past pallet range


def test_pallet_to_pick_from_segment_c():
    # Robot at (15..24, 1) should pick from (x, 0)
    assert pallet_to_pick_from(15, 1, 15) == (15, 0)
    assert pallet_to_pick_from(24, 1, 15) == (24, 0)
    assert pallet_to_pick_from(14, 1, 15) is None   # x=14 has no pallet


def test_pallet_to_pick_from_other_segments():
    # Segments not adjacent to pallets
    assert pallet_to_pick_from(26, 3, 15) is None
    assert pallet_to_pick_from(26, 1, 15) is None
    assert pallet_to_pick_from(14, 0, 15) is None


def test_second_nest_track():
    track = build_conveyor_track(35)
    assert track[0] == (35, 3)
    assert (34, 0) in track   # fulfill point for nest_x=35


def test_all_track_positions_in_grid():
    track = build_conveyor_track(15)
    for x, y in track:
        assert 0 <= x < 60, f"x={x} out of bounds"
        assert 0 <= y < 40, f"y={y} out of bounds"


def test_nest_robot_assignment():
    from jr_walker.entities import Robot
    from jr_walker.nest_builder import assign_robots_to_nests

    robots = [
        Robot(id=0, nest_id=-1, x=58, y=13),
        Robot(id=1, nest_id=-1, x=52, y=5),
        Robot(id=2, nest_id=-1, x=22, y=16),
        Robot(id=3, nest_id=-1, x=41, y=9),
        Robot(id=4, nest_id=-1, x=41, y=14),
    ]
    assign_robots_to_nests(robots, [15, 35])

    # Every robot should be assigned
    for r in robots:
        assert r.nest_id in (15, 35)

    # Each nest gets at least 1 robot
    nest_15 = [r for r in robots if r.nest_id == 15]
    nest_35 = [r for r in robots if r.nest_id == 35]
    assert len(nest_15) >= 1
    assert len(nest_35) >= 1
    assert len(nest_15) + len(nest_35) == 5
