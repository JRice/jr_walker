import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jr_walker.hierarchical import MiniBoxMotionPlanner, SetupTaskPlanner  # noqa: E402
from jr_walker.logic import DockSuggestion, OrderSuggestion, RelocateSuggestion, SetupSuggestion  # noqa: E402
from types import SimpleNamespace
import collections


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

    def test_suggestion_string_formats(self) -> None:
        setup = SetupSuggestion(
            SimpleNamespace(
                source_xy=(1, 2),
                target_xy=(3, 4),
            )
        )
        setup.assigned_robot_id = 7
        self.assertEqual(
            str(setup),
            "SetupSuggestion: source (1,2) to dest (3,4) for robot 7",
        )

        reloc_job = SimpleNamespace(
            hotspot=(10, 10),
            placement_offset=(0, 0),
            preferred_target_xy=(12, 13),
        )
        reloc = RelocateSuggestion(reloc_job, scheduler=SimpleNamespace(pallet_cells_for_sku=lambda _sku: []))
        reloc.job.sku = 1
        self.assertEqual(
            str(reloc),
            "RelocateSuggestion: source (10,10) to dest (12,13)",
        )

        order = OrderSuggestion(
            order_idx=0,
            order=collections.Counter({3: 1, 1: 2}),
            cluster={1: (1, 1), 3: (2, 2)},
            order_gain_constant=100.0,
            warehouse_width=60,
            warehouse_height=40,
            scheduler=SimpleNamespace(pallets={}),
        )
        self.assertEqual(str(order), "OrderSuggestion: order (2x1 1x3)")

        dock = DockSuggestion(sku=5, plan=[4, 8, 9], gain=1.0, pallet_xy=(0, 0))
        self.assertEqual(
            str(dock),
            "DockSuggestion: robot <R> plans 3 orders with SKU 5: (4 8 9)",
        )


if __name__ == "__main__":
    unittest.main()
