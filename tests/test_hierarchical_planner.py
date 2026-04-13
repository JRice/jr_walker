import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jr_walker.hierarchical import MiniBoxMotionPlanner, SetupTaskPlanner  # noqa: E402


class HierarchicalPlannerTests(unittest.TestCase):
    def test_setup_task_planner_emits_macro_sequence(self) -> None:
        planner = SetupTaskPlanner()
        macros = planner.build_setup_relocation_macros(
            stand_xy=(10, 10),
            source_xy=(10, 11),
            target_xy=(20, 2),
            requires_local_maneuver=True,
        )
        self.assertEqual(
            [m.name for m in macros],
            ["move_to_stand", "dock_source", "maneuver_pivot", "carry_to_target", "undock_target"],
        )

    def test_mini_box_pivot_builds_primitive_steps(self) -> None:
        motion = MiniBoxMotionPlanner(width=60, height=40, box_radius=2)
        plan = motion.plan_pivot(
            robot_xy=(5, 5),
            pallet_xy=(5, 4),
            start_offset=(0, -1),
            target_offset=(-1, 0),
            static_blocked_cells=[],
        )
        self.assertIsNotNone(plan)
        steps = list(plan.steps)
        self.assertEqual(steps[0].action, "undock")
        self.assertEqual(steps[-1].action, "dock")
        move_targets = [(s.x, s.y) for s in steps if s.action == "move"]
        self.assertEqual(move_targets[-1], (6, 4))

    def test_mini_box_walk_obeys_local_bounds(self) -> None:
        motion = MiniBoxMotionPlanner(width=60, height=40, box_radius=2)
        path = motion.plan_local_walk(
            start_xy=(5, 5),
            goal_xy=(9, 5),  # outside 5x5 window around start (x in [3,7])
            pallet_xy=(5, 4),
            static_blocked_cells=[],
        )
        self.assertIsNone(path)

    def test_mini_box_walk_detours_around_blockers(self) -> None:
        motion = MiniBoxMotionPlanner(width=60, height=40, box_radius=2)
        path = motion.plan_local_walk(
            start_xy=(5, 5),
            goal_xy=(6, 4),
            pallet_xy=(5, 4),
            static_blocked_cells=[(6, 5)],
        )
        self.assertIsNotNone(path)
        assert path is not None
        self.assertNotIn((6, 5), path)
        self.assertEqual(path[-1], (6, 4))


if __name__ == "__main__":
    unittest.main()
