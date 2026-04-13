import sqlite3
import sys
import tempfile
import unittest
import collections
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from jr_walker.solver import RelocationJob, SetupJob, WarehouseSolver, PastRunAnalysis, _select_best_non_test_run_id  # noqa: E402
    from jr_walker.logic import SetupSuggestion  # noqa: E402
except Exception:  # pragma: no cover - environment-dependent optional import
    RelocationJob = None
    SetupJob = None
    SetupSuggestion = None
    WarehouseSolver = None
    PastRunAnalysis = None
    _select_best_non_test_run_id = None


@unittest.skipIf(
    WarehouseSolver is None or RelocationJob is None or SetupJob is None or SetupSuggestion is None,
    "solver dependencies unavailable",
)
class SolverMetadataPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_roots: list[tempfile.TemporaryDirectory[str]] = []

    def tearDown(self) -> None:
        for tmp in self._tmp_roots:
            tmp.cleanup()
        self._tmp_roots.clear()

    def _workspace_case_dir(self) -> Path:
        base = ROOT / "output"
        base.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.TemporaryDirectory(prefix="test_case_", dir=base)
        self._tmp_roots.append(tmp)
        case_dir = Path(tmp.name)
        return case_dir

    def _new_solver_shell(self) -> WarehouseSolver:
        solver = WarehouseSolver.__new__(WarehouseSolver)
        solver.state = SimpleNamespace(
            width=60,
            height=40,
            robots=[(0, 0)],
            pallets={(5, 5): 1},
            orders=[[1]],
        )
        solver.scheduler = SimpleNamespace(pallets={})
        solver.travel_lane_cells = set()
        solver.past_analysis = PastRunAnalysis()
        solver.config = SimpleNamespace(
            relocation_edge_band=6,
            relocate_chunk_size=1,
            dispatch_validate_every_makespan=500,
            strict_no_swap=False,
            astar_slow_ms=40.0,
            astar_print_slow=False,
            astar_log_blocked=False,
        )
        return solver

    def test_periodic_dispatch_validation_runs_at_interval(self) -> None:
        solver = self._new_solver_shell()
        solver.config = SimpleNamespace(dispatch_validate_every_makespan=500, strict_no_swap=False)
        solver._next_dispatch_validation_makespan = 500
        solver.robots = [SimpleNamespace(last_t=500)]
        solver.actions = SimpleNamespace(sorted_actions=lambda: [(0, 0, "move", 1, 0)])
        solver._log = lambda msg: None

        solver._maybe_validate_dispatch_progress_or_raise(
            completed=10,
            total_orders=100,
            dispatch_count=25,
        )
        self.assertEqual(solver._next_dispatch_validation_makespan, 1000)

    def test_periodic_dispatch_validation_raises_on_invalid_prefix(self) -> None:
        solver = self._new_solver_shell()
        solver.config = SimpleNamespace(dispatch_validate_every_makespan=500, strict_no_swap=False)
        solver._next_dispatch_validation_makespan = 500
        solver.robots = [SimpleNamespace(last_t=700)]
        solver.actions = SimpleNamespace(sorted_actions=lambda: [(0, 0, "move", 2, 0)])
        solver._log = lambda msg: None

        with self.assertRaises(RuntimeError):
            solver._maybe_validate_dispatch_progress_or_raise(
                completed=12,
                total_orders=100,
                dispatch_count=30,
            )

    def test_select_best_non_test_run_id_prefers_best_non_test(self) -> None:
        db_path = self._workspace_case_dir() / "meta.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "CREATE TABLE metadata_runs (run_id INTEGER, solution_path TEXT, makespan INTEGER)"
            )
            conn.executemany(
                "INSERT INTO metadata_runs (run_id, solution_path, makespan) VALUES (?, ?, ?)",
                [
                    (1, "output/stride_10_solution_100.txt", 100),
                    (2, "output/solution_130.txt", 130),
                    (3, "output/solution_120.txt", 120),
                    (4, "output/partial_solution_90.txt", 90),
                ],
            )
            selected = _select_best_non_test_run_id(conn)
        finally:
            conn.close()
        self.assertEqual(selected, 3)

    def test_select_best_non_test_run_id_falls_back_to_best_any(self) -> None:
        db_path = self._workspace_case_dir() / "meta.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "CREATE TABLE metadata_runs (run_id INTEGER, solution_path TEXT, makespan INTEGER)"
            )
            conn.executemany(
                "INSERT INTO metadata_runs (run_id, solution_path, makespan) VALUES (?, ?, ?)",
                [
                    (1, "output/stride_10_solution_100.txt", 100),
                    (2, "output/partial_solution_80.txt", 80),
                ],
            )
            selected = _select_best_non_test_run_id(conn)
        finally:
            conn.close()
        self.assertEqual(selected, 2)

    def test_choose_metadata_guided_target_prefers_low_use_and_avoids_lane(self) -> None:
        solver = self._new_solver_shell()
        solver.state = SimpleNamespace(width=3, height=3)
        solver.travel_lane_cells = {(1, 1)}
        solver.past_analysis.high_use_cells = {(0, 1)}
        solver.past_analysis.sku_cells = {7: [(1, 1, 50)]}
        solver.past_analysis.use_by_cell = {
            (1, 0): 20,
            (2, 1): 5,
            (1, 2): 1,
            (0, 0): 50,
            (2, 0): 50,
            (0, 2): 50,
            (2, 2): 50,
        }
        job = RelocationJob(sku=7, bucket="top_x0_29", hotspot=(1, 1), score=1.0)

        target = solver._choose_metadata_guided_relocation_target(
            job=job,
            reserved_targets=set(),
        )
        self.assertEqual(target, (1, 2))

    def test_choose_metadata_guided_target_respects_edge_band(self) -> None:
        solver = self._new_solver_shell()
        solver.state = SimpleNamespace(width=20, height=20)
        solver.config = SimpleNamespace(relocation_edge_band=2)
        solver.travel_lane_cells = set()
        solver.past_analysis.high_use_cells = set()
        solver.past_analysis.sku_cells = {7: [(10, 10, 50)]}
        solver.past_analysis.use_by_cell = {(10, 10): 0}
        job = RelocationJob(sku=7, bucket="top_x0_29", hotspot=(10, 10), score=1.0)

        target = solver._choose_metadata_guided_relocation_target(
            job=job,
            reserved_targets=set(),
        )
        self.assertIsNotNone(target)
        tx, ty = target
        edge_dist = min(tx, solver.state.width - 1 - tx, ty, solver.state.height - 1 - ty)
        self.assertLessEqual(edge_dist, 2)

    def test_build_travel_lane_cells_includes_metadata_high_use_halo(self) -> None:
        solver = self._new_solver_shell()
        solver.past_analysis.high_use_cells = {(15, 15)}

        lanes = solver._build_travel_lane_cells(lane_width=0)
        self.assertIn((15, 15), lanes)
        self.assertIn((15, 14), lanes)
        self.assertIn((16, 15), lanes)

    def test_setup_inward_step_prefers_nearest_edge_direction(self) -> None:
        solver = self._new_solver_shell()
        solver.state = SimpleNamespace(width=60, height=40)

        self.assertEqual(solver._setup_inward_step((20, 5)), -1)
        self.assertEqual(solver._setup_inward_step((20, 35)), 1)

    def test_setup_pull_directions_prioritize_larger_axis(self) -> None:
        solver = self._new_solver_shell()

        self.assertEqual(
            solver._setup_pull_directions((10, 10), (14, 11)),
            [(1, 0), (0, 1)],
        )
        self.assertEqual(
            solver._setup_pull_directions((10, 10), (9, 4)),
            [(0, -1), (-1, 0)],
        )

    def test_setup_target_candidates_same_column_frontier_only(self) -> None:
        solver = self._new_solver_shell()
        solver.state = SimpleNamespace(width=20, height=20)
        solver.scheduler = SimpleNamespace(pallets={})
        job = SetupJob(
            sku=1,
            hotspot=(10, 5),
            source_pallet_id=7,
            source_xy=(3, 3),
            target_xy=(10, 5),
        )

        cells = solver._setup_target_candidates(job, source_xy=(3, 3), limit=6)
        self.assertTrue(cells)
        self.assertTrue(all(x == 10 for x, _ in cells))
        self.assertTrue(all(y <= 5 for _, y in cells))

    def test_candidate_robots_for_setup_prefers_reachable_probe(self) -> None:
        solver = self._new_solver_shell()
        solver.config = SimpleNamespace(max_robots_per_suggestion=3, path_step_limit=50)
        solver.robots = [
            SimpleNamespace(id=0, x=50, y=50, last_t=0),
            SimpleNamespace(id=1, x=1, y=1, last_t=0),
        ]
        job = SetupJob(
            sku=1,
            hotspot=(10, 5),
            source_pallet_id=7,
            source_xy=(3, 3),
            target_xy=(10, 5),
        )
        suggestion = SetupSuggestion(job)

        # Make robot 0 unreachable and robot 1 reachable regardless of distance.
        solver._setup_probe_robot_for_job = lambda robot, _job: (1, 99, robot.last_t, robot.id) if robot.id == 0 else (0, 5, robot.last_t, robot.id)

        ranked = solver._candidate_robots_for_suggestion(suggestion)
        self.assertEqual(ranked[0].id, 1)

    def test_candidate_robots_for_setup_uses_assigned_hotspot_robot(self) -> None:
        solver = self._new_solver_shell()
        solver.config = SimpleNamespace(max_robots_per_suggestion=3, path_step_limit=50)
        solver.robots = [
            SimpleNamespace(id=0, x=10, y=10, last_t=0),
            SimpleNamespace(id=1, x=20, y=20, last_t=0),
            SimpleNamespace(id=2, x=30, y=30, last_t=0),
        ]
        solver.setup_jobs = [object()]
        solver._completed_setup_pallet_ids = set()
        solver._setup_robot_by_hotspot = {(10, 5): 2}
        solver._setup_robot_ids = {2}

        job = SetupJob(
            sku=1,
            hotspot=(10, 5),
            source_pallet_id=7,
            source_xy=(3, 3),
            target_xy=(10, 5),
        )
        suggestion = SetupSuggestion(job)

        ranked = solver._candidate_robots_for_suggestion(suggestion)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].id, 2)

    def test_non_setup_suggestions_exclude_setup_robots_during_setup_phase(self) -> None:
        solver = self._new_solver_shell()
        solver.config = SimpleNamespace(max_robots_per_suggestion=3, path_step_limit=50)
        solver.robots = [
            SimpleNamespace(id=0, x=0, y=0, last_t=0),
            SimpleNamespace(id=1, x=5, y=5, last_t=0),
            SimpleNamespace(id=2, x=9, y=9, last_t=0),
        ]
        solver.setup_jobs = [
            SetupJob(sku=1, hotspot=(10, 5), source_pallet_id=70, source_xy=(3, 3), target_xy=(10, 5)),
            SetupJob(sku=2, hotspot=(10, 15), source_pallet_id=71, source_xy=(4, 4), target_xy=(10, 15)),
        ]
        solver._completed_setup_pallet_ids = set()
        solver._dropped_setup_pallet_ids = set()
        solver._setup_robot_by_hotspot = {(10, 5): 0, (10, 15): 1}
        solver._setup_robot_ids = {0, 1}

        suggestion = SimpleNamespace(center=(8, 8))
        ranked = solver._candidate_robots_for_suggestion(suggestion)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].id, 2)

    def test_non_setup_suggestions_release_robot_when_hotspot_done(self) -> None:
        solver = self._new_solver_shell()
        solver.config = SimpleNamespace(max_robots_per_suggestion=3, path_step_limit=50)
        solver.robots = [
            SimpleNamespace(id=0, x=0, y=0, last_t=0),
            SimpleNamespace(id=1, x=5, y=5, last_t=0),
            SimpleNamespace(id=2, x=9, y=9, last_t=0),
        ]
        solver.setup_jobs = [
            SetupJob(sku=1, hotspot=(10, 5), source_pallet_id=70, source_xy=(3, 3), target_xy=(10, 5)),
            SetupJob(sku=2, hotspot=(10, 15), source_pallet_id=71, source_xy=(4, 4), target_xy=(10, 15)),
        ]
        # Robot 0 has finished its hotspot's setup work; robot 1 still has pending setup.
        solver._completed_setup_pallet_ids = {70}
        solver._dropped_setup_pallet_ids = set()
        solver._setup_robot_by_hotspot = {(10, 5): 0, (10, 15): 1}
        solver._setup_robot_ids = {0, 1}

        suggestion = SimpleNamespace(center=(8, 8))
        ranked = solver._candidate_robots_for_suggestion(suggestion)
        ranked_ids = {r.id for r in ranked}

        self.assertIn(0, ranked_ids)
        self.assertIn(2, ranked_ids)
        self.assertNotIn(1, ranked_ids)

    def test_iter_sku_anchor_rows_groups_counts_by_chunk(self) -> None:
        solver = self._new_solver_shell()
        solver.state = SimpleNamespace(width=12, height=12)
        solver.config = SimpleNamespace(relocation_edge_band=20, relocate_chunk_size=3)
        solver.past_analysis.sku_cells = {
            7: [
                (0, 0, 10),
                (2, 1, 5),
                (5, 5, 4),
            ]
        }
        solver.past_analysis.use_by_cell = {}

        rows = solver._iter_sku_anchor_rows(7, limit=5)
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(rows[0][2], 15)
        self.assertLessEqual(rows[0][0], 2)
        self.assertLessEqual(rows[0][1], 2)

    def test_strict_no_swap_rejects_robot_position_swap(self) -> None:
        solver = self._new_solver_shell()
        solver.state = SimpleNamespace(
            width=60,
            height=40,
            robots=[(0, 0), (1, 0)],
            pallets={(5, 5): 1},
            orders=[[1]],
        )
        solver.config = SimpleNamespace(strict_no_swap=True)

        actions = [
            (0, 0, "move", 1, 0),
            (0, 1, "move", 0, 0),
        ]
        ok = solver._validate_candidate_actions(actions, require_complete=False)
        self.assertFalse(ok)

    def test_dispatch_raises_when_full_queue_scan_has_no_progress(self) -> None:
        solver = self._new_solver_shell()
        solver.orders = [SimpleNamespace(order_idx=0, order=collections.Counter({1: 1}))]
        solver.robots = [SimpleNamespace(id=0, x=0, y=0, last_t=0)]
        solver.setup_jobs = []
        solver._completed_setup_pallet_ids = set()
        solver._dropped_setup_pallet_ids = set()
        solver.relocated_skus = set()
        solver.docked_skus = set()
        solver._suggestion_fail_counts = {}
        solver._suggestion_backoff_until_cycle = {}
        solver._robot_fail_streak = collections.defaultdict(int)
        solver._dispatch_cycle = 0
        solver._parking_moves = 0
        solver.actions = SimpleNamespace(sorted_actions=lambda: [])
        solver._astar_calls = 0
        solver._log = lambda msg: None

        solver.config = SimpleNamespace(
            num_allowed_relocations=0,
            robot_fail_streak_for_parking=999,
            suggestion_retry_limit=2,
            suggestion_backoff_base_cycles=1,
            suggestion_backoff_max_cycles=2,
            max_robots_per_suggestion=1,
            progress_every=1000,
        )

        stuck = SimpleNamespace(sku=1, center=(0, 0), expected_gain=1.0, score=lambda: 1.0)
        solver._build_suggestion_queue = lambda: [stuck]
        solver._check_global_limits_or_raise = lambda _: None
        solver._log_solve_start = lambda _: None
        solver._suggestion_key = lambda _: "dock:1:0:0"
        solver._candidate_robots_for_suggestion = lambda _: list(solver.robots)
        solver._plan_idle_parking_move = lambda _: False
        solver._suggestion_backoff_cycles = lambda _: 1
        solver._repair_idle_wait_conflicts = lambda actions: actions

        # Treat our synthetic suggestion as dock flow and force failure.
        from jr_walker.logic import DockSuggestion  # local import for test-only monkeypatch

        real_dock_suggestion = DockSuggestion(sku=1, plan=[], gain=1.0, pallet_xy=(0, 0))
        solver._build_suggestion_queue = lambda: [real_dock_suggestion]
        solver._plan_dock_pallet = lambda _robot, _sku: False

        captured_logs: list[str] = []
        solver._log = lambda msg: captured_logs.append(str(msg))

        with self.assertRaises(RuntimeError):
            solver._find_solution_actions_core()

        self.assertTrue(any("dispatcher_stall_robots" in msg for msg in captured_logs))
        self.assertTrue(any("dispatcher_stall_queue[0] DockSuggestion:" in msg for msg in captured_logs))

    def test_build_setup_jobs_left_edge_orders_odds_then_evens_inward(self) -> None:
        solver = self._new_solver_shell()
        solver.state = SimpleNamespace(width=60, height=40)
        solver.config = SimpleNamespace(setup_hotspots=[(0, 15)])
        solver._log = lambda msg: None

        pallets = {}
        for sku in range(1, 21):
            pallets[(10 + ((sku - 1) % 5), 5 + ((sku - 1) // 5))] = sku
        sku_cells = collections.defaultdict(list)
        for cell, sku in pallets.items():
            sku_cells[int(sku)].append(cell)
        solver.scheduler = SimpleNamespace(
            pallets=pallets,
            pallet_cells_for_sku=lambda sku: list(sku_cells.get(int(sku), [])),
        )
        solver.pallet_id_by_coord = {cell: idx for idx, cell in enumerate(pallets.keys())}

        jobs = solver._build_setup_jobs()
        self.assertEqual(len(jobs), 20)

        expected_skus = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.assertEqual([job.sku for job in jobs], expected_skus)

        expected_targets = (
            [(0, 15 + i) for i in range(10)]
            + [(2, 15 + i) for i in range(10)]
        )
        self.assertEqual([job.target_xy for job in jobs], expected_targets)

    def test_build_setup_jobs_top_edge_orders_odds_then_evens_inward(self) -> None:
        solver = self._new_solver_shell()
        solver.state = SimpleNamespace(width=60, height=40)
        solver.config = SimpleNamespace(setup_hotspots=[(20, 0)])
        solver._log = lambda msg: None

        pallets = {}
        for sku in range(1, 21):
            pallets[(10 + ((sku - 1) % 5), 5 + ((sku - 1) // 5))] = sku
        sku_cells = collections.defaultdict(list)
        for cell, sku in pallets.items():
            sku_cells[int(sku)].append(cell)
        solver.scheduler = SimpleNamespace(
            pallets=pallets,
            pallet_cells_for_sku=lambda sku: list(sku_cells.get(int(sku), [])),
        )
        solver.pallet_id_by_coord = {cell: idx for idx, cell in enumerate(pallets.keys())}

        jobs = solver._build_setup_jobs()
        self.assertEqual(len(jobs), 20)

        expected_skus = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.assertEqual([job.sku for job in jobs], expected_skus)

        expected_targets = (
            [(20 + i, 0) for i in range(10)]
            + [(20 + i, 2) for i in range(10)]
        )
        self.assertEqual([job.target_xy for job in jobs], expected_targets)

    def test_build_setup_jobs_projects_non_edge_hotspot_to_edge_template(self) -> None:
        solver = self._new_solver_shell()
        solver.state = SimpleNamespace(width=60, height=40)
        solver.config = SimpleNamespace(setup_hotspots=[(22, 10)])
        solver._log = lambda msg: None

        pallets = {}
        for sku in range(1, 21):
            pallets[(10 + ((sku - 1) % 5), 5 + ((sku - 1) // 5))] = sku
        sku_cells = collections.defaultdict(list)
        for cell, sku in pallets.items():
            sku_cells[int(sku)].append(cell)
        solver.scheduler = SimpleNamespace(
            pallets=pallets,
            pallet_cells_for_sku=lambda sku: list(sku_cells.get(int(sku), [])),
        )
        solver.pallet_id_by_coord = {cell: idx for idx, cell in enumerate(pallets.keys())}

        jobs = solver._build_setup_jobs()
        self.assertEqual(len(jobs), 20)
        # (22,10) projects to the nearest edge anchor (22,0).
        self.assertTrue(all(job.hotspot == (22, 0) for job in jobs))
        expected_targets = (
            [(22 + i, 0) for i in range(10)]
            + [(22 + i, 2) for i in range(10)]
        )
        self.assertEqual([job.target_xy for job in jobs], expected_targets)


if __name__ == "__main__":
    unittest.main()
