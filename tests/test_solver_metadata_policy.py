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

    def test_nearest_unreserved_setup_source_prefers_local_owner(self) -> None:
        solver = self._new_solver_shell()
        solver.scheduler = SimpleNamespace(
            pallet_cells_for_sku=lambda _sku: [(10, 10), (1, 1)],
        )
        solver.pallet_id_by_coord = {
            (10, 10): 101,
            (1, 1): 202,
        }
        hotspot = (20, 20)
        owner_map = {
            101: (0, 0),
            202: hotspot,
        }

        source = solver._nearest_unreserved_pallet_for_sku(
            sku=7,
            hotspot=hotspot,
            reserved_pallet_ids=set(),
            source_owner_by_pallet_id=owner_map,
        )
        self.assertEqual(source, ((1, 1), 202))

    def test_setup_stand_priority_avoids_foreign_active_corridors(self) -> None:
        solver = self._new_solver_shell()
        solver._completed_setup_pallet_ids = set()
        solver._dropped_setup_pallet_ids = set()
        solver._setup_jobs_by_hotspot = {
            (0, 0): [SimpleNamespace(source_pallet_id=1)],
            (5, 0): [SimpleNamespace(source_pallet_id=2)],
        }
        solver._setup_slot_candidates = lambda hotspot, limit=80: (
            [(5, 0), (5, 1)] if hotspot == (5, 0) else [(0, 0), (0, 1)]
        )
        stand_cells = [(5, 0), (4, 0), (5, 1)]

        ordered, penalized, corridor_size = solver._prioritize_setup_stands_away_from_foreign_corridors(
            stand_cells,
            (0, 0),
        )
        self.assertEqual(ordered, [(4, 0), (5, 0), (5, 1)])
        self.assertEqual(penalized, 2)
        self.assertEqual(corridor_size, 2)

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

    def test_dispatch_no_progress_realistic_mode_attempts_corner_retire(self) -> None:
        solver = self._new_solver_shell()
        solver.orders = [SimpleNamespace(order_idx=0, order=collections.Counter({1: 1}))]
        solver.robots = [SimpleNamespace(id=0, x=0, y=0, last_t=0)]
        solver.setup_jobs = []
        solver._completed_setup_pallet_ids = set()
        solver._dropped_setup_pallet_ids = set()
        solver._retired_robot_ids = set()
        solver.relocated_skus = set()
        solver.docked_skus = set()
        solver._suggestion_fail_counts = {}
        solver._suggestion_backoff_until_cycle = {}
        solver._robot_fail_streak = collections.defaultdict(int)
        solver._dispatch_cycle = 0
        solver._parking_moves = 0
        solver.actions = SimpleNamespace(sorted_actions=lambda: [])
        solver._astar_calls = 0

        solver.config = SimpleNamespace(
            num_allowed_relocations=0,
            robot_fail_streak_for_parking=999,
            suggestion_retry_limit=2,
            suggestion_backoff_base_cycles=1,
            suggestion_backoff_max_cycles=2,
            max_robots_per_suggestion=1,
            progress_every=1000,
            realistic_fail_mode=True,
            order_stagnation_cycle_limit=4,
        )

        from jr_walker.logic import DockSuggestion

        real_dock_suggestion = DockSuggestion(sku=1, plan=[], gain=1.0, pallet_xy=(0, 0))
        solver._build_suggestion_queue = lambda: [real_dock_suggestion]
        solver._check_global_limits_or_raise = lambda _: None
        solver._log_solve_start = lambda _: None
        solver._suggestion_key = lambda _: "dock:1:0:0"
        solver._candidate_robots_for_suggestion = lambda _: list(solver.robots)
        solver._plan_idle_parking_move = lambda _: False
        solver._suggestion_backoff_cycles = lambda _: 1
        solver._repair_idle_wait_conflicts = lambda actions: actions
        solver._plan_dock_pallet = lambda _robot, _sku: False

        retire_calls = {"count": 0}

        def _retire(_robot):
            retire_calls["count"] += 1
            solver._retired_robot_ids.add(0)
            return True

        solver._plan_retire_robot_to_corner = _retire
        solver._log = lambda _msg: None

        with self.assertRaises(RuntimeError):
            solver._find_solution_actions_core()
        self.assertGreaterEqual(retire_calls["count"], 1)

    def test_dock_suggestion_reserve_footprint_conflict_is_skipped(self) -> None:
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
        solver._setup_retry_not_before_timestep = {}
        solver._setup_wait_logged_cycle = {}
        solver._robot_fail_streak = collections.defaultdict(int)
        solver._dispatch_cycle = 0
        solver._parking_moves = 0
        solver.actions = SimpleNamespace(sorted_actions=lambda: [])
        solver._astar_calls = 0
        solver._active_suggestion = None
        solver._active_robot_id = None
        solver.config = SimpleNamespace(
            num_allowed_relocations=0,
            robot_fail_streak_for_parking=999,
            suggestion_retry_limit=2,
            suggestion_backoff_base_cycles=1,
            suggestion_backoff_max_cycles=2,
            setup_retry_wait_ticks=0,
            max_robots_per_suggestion=1,
            progress_every=1000,
            realistic_fail_mode=False,
            order_stagnation_cycle_limit=8,
        )

        from jr_walker.logic import DockSuggestion

        real_dock_suggestion = DockSuggestion(sku=1, plan=[], gain=1.0, pallet_xy=(0, 0))
        solver._build_suggestion_queue = lambda: [real_dock_suggestion]
        solver._check_global_limits_or_raise = lambda _: None
        solver._log_solve_start = lambda _: None
        solver._suggestion_key = lambda _: "dock:1:0:0"
        solver._candidate_robots_for_suggestion = lambda _: list(solver.robots)
        solver._plan_idle_parking_move = lambda _: False
        solver._suggestion_backoff_cycles = lambda _: 1
        solver._repair_idle_wait_conflicts = lambda actions: actions
        solver._validate_candidate_actions = lambda _a, log_on_error=False: True
        solver._plan_dock_pallet = lambda _robot, _sku: (_ for _ in ()).throw(
            ValueError("Cannot reserve footprint for robot=3 at t=711, x=16, y=1")
        )

        captured_logs: list[str] = []
        solver._log = lambda msg: captured_logs.append(str(msg))

        out = solver._find_solution_actions_core()
        self.assertIsInstance(out, list)
        self.assertTrue(any("dock_suggestion_skipped" in msg for msg in captured_logs))

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

    def test_candidate_fulfill_cells_avoids_pallet_perimeter_cells(self) -> None:
        solver = self._new_solver_shell()
        solver.state = SimpleNamespace(width=6, height=4)
        solver.scheduler = SimpleNamespace(pallets={(0, 0): 1, (3, 0): 2, (5, 3): 3})
        solver.robots = [SimpleNamespace(id=0, x=2, y=1, last_t=0)]

        cells = solver._candidate_fulfill_cells(from_xy=(2, 1), limit=20)
        self.assertNotIn((0, 0), cells)
        self.assertNotIn((3, 0), cells)
        self.assertNotIn((5, 3), cells)
        self.assertIn((2, 0), cells)

    def test_plan_order_uses_alternate_fulfill_cell_when_first_rejected(self) -> None:
        solver = self._new_solver_shell()
        solver.state = SimpleNamespace(width=10, height=6)
        robot = SimpleNamespace(
            id=0,
            x=0,
            y=1,
            last_t=0,
            storage=collections.Counter(),
            docks={},
        )
        solver.robots = [robot]
        solver.scheduler = SimpleNamespace(
            pallets={},
            candidate_pick_options=lambda _remaining, _xy: [],
        )
        solver.planner = SimpleNamespace(can_occupy=lambda *_args, **_kwargs: True)
        solver._candidate_fulfill_cells = lambda **_kwargs: [(1, 0), (2, 0)]
        solver._safe_plan_path = lambda r, x, y: [(r.last_t + 1, x, y)] if (r.x != x or r.y != y) else []

        def _can_commit(actions):
            # Reject plans that fulfill at (1,0), accept others.
            return not any(a[2] == "fulfill" and a[3] == 1 and a[4] == 0 for a in actions)

        solver._can_commit_pending_actions = _can_commit
        captured: dict = {}

        def _capture_commit(**kwargs):
            captured["pending_actions"] = list(kwargs["pending_actions"])

        solver._commit_plan = _capture_commit

        ok = solver._plan_order_for_robot(0, collections.Counter(), robot)
        self.assertTrue(ok)
        fulfills = [a for a in captured["pending_actions"] if a[2] == "fulfill"]
        self.assertEqual(len(fulfills), 1)
        self.assertEqual((fulfills[0][3], fulfills[0][4]), (2, 0))

    def test_plan_order_prefers_cluster_pallet_cell_first(self) -> None:
        solver = self._new_solver_shell()
        solver.state = SimpleNamespace(width=12, height=12)
        robot = SimpleNamespace(
            id=0,
            x=0,
            y=0,
            last_t=0,
            storage=collections.Counter(),
            docks={},
        )
        solver.robots = [robot]
        solver.scheduler = SimpleNamespace(
            pallets={(2, 2): 7, (9, 9): 7},
            pick_cells_for_pallet=lambda xy: [(xy[0], max(0, xy[1] - 1))],
            candidate_pick_options=lambda _remaining, _xy: [
                (0, 7, (2, 2), (2, 1)),
                (0, 7, (9, 9), (9, 8)),
            ],
        )
        solver.planner = SimpleNamespace(can_occupy=lambda *_args, **_kwargs: True)
        solver._safe_plan_path = lambda r, x, y: [(r.last_t + 1, x, y)] if (r.x != x or r.y != y) else []
        solver._is_pick_target_static_at_time = lambda *_args, **_kwargs: True
        solver._candidate_fulfill_cells = lambda **_kwargs: [(0, 0)]
        solver._can_commit_pending_actions = lambda _actions: True
        captured: dict = {}
        solver._commit_plan = lambda **kwargs: captured.update({"pending_actions": list(kwargs["pending_actions"])})

        ok = solver._plan_order_for_robot(
            0,
            collections.Counter({7: 1}),
            robot,
            preferred_pallet_cells_by_sku={7: [(9, 9)]},
        )
        self.assertTrue(ok)
        picks = [a for a in captured["pending_actions"] if a[2] == "pick"]
        self.assertEqual(len(picks), 1)
        self.assertEqual((picks[0][3], picks[0][4]), (9, 9))

    def test_plan_order_falls_back_when_preferred_unavailable(self) -> None:
        solver = self._new_solver_shell()
        solver.state = SimpleNamespace(width=12, height=12)
        robot = SimpleNamespace(
            id=0,
            x=0,
            y=0,
            last_t=0,
            storage=collections.Counter(),
            docks={},
        )
        solver.robots = [robot]
        solver.scheduler = SimpleNamespace(
            pallets={(2, 2): 7},
            pick_cells_for_pallet=lambda xy: [(xy[0], max(0, xy[1] - 1))],
            candidate_pick_options=lambda _remaining, _xy: [
                (0, 7, (2, 2), (2, 1)),
            ],
        )
        solver.planner = SimpleNamespace(can_occupy=lambda *_args, **_kwargs: True)
        solver._safe_plan_path = lambda r, x, y: [(r.last_t + 1, x, y)] if (r.x != x or r.y != y) else []
        solver._is_pick_target_static_at_time = lambda *_args, **_kwargs: True
        solver._candidate_fulfill_cells = lambda **_kwargs: [(0, 0)]
        solver._can_commit_pending_actions = lambda _actions: True
        captured: dict = {}
        solver._commit_plan = lambda **kwargs: captured.update({"pending_actions": list(kwargs["pending_actions"])})

        ok = solver._plan_order_for_robot(
            0,
            collections.Counter({7: 1}),
            robot,
            preferred_pallet_cells_by_sku={7: [(9, 9)]},
        )
        self.assertTrue(ok)
        picks = [a for a in captured["pending_actions"] if a[2] == "pick"]
        self.assertEqual(len(picks), 1)
        self.assertEqual((picks[0][3], picks[0][4]), (2, 2))

    def test_plan_order_bfs_falls_back_to_global_after_local_radius(self) -> None:
        solver = self._new_solver_shell()
        solver.state = SimpleNamespace(width=12, height=12)
        solver.config = SimpleNamespace(
            order_pick_local_manhattan_radius=1,
            order_other_hotspot_penalty=0.0,
            order_other_hotspot_penalty_radius=6,
            setup_hotspots=[(0, 0), (11, 11)],
        )
        robot = SimpleNamespace(
            id=0,
            x=0,
            y=0,
            last_t=0,
            storage=collections.Counter(),
            docks={},
            assigned_hotspot=(0, 0),
        )
        solver.robots = [robot]
        solver.scheduler = SimpleNamespace(
            pallets={(8, 8): 7},
            pick_cells_for_pallet=lambda xy: [(xy[0], xy[1] - 1)],
        )
        solver.planner = SimpleNamespace(can_occupy=lambda *_args, **_kwargs: True)
        solver._safe_plan_path = lambda r, x, y: [(r.last_t + 1, x, y)] if (r.x != x or r.y != y) else []
        solver._is_pick_target_static_at_time = lambda *_args, **_kwargs: True
        solver._candidate_fulfill_cells = lambda **_kwargs: [(0, 0)]
        solver._can_commit_pending_actions = lambda _actions: True
        solver._log = lambda *_args, **_kwargs: None
        captured: dict = {}
        solver._commit_plan = lambda **kwargs: captured.update({"pending_actions": list(kwargs["pending_actions"])})

        ok = solver._plan_order_for_robot(0, collections.Counter({7: 1}), robot)
        self.assertTrue(ok)
        picks = [a for a in captured["pending_actions"] if a[2] == "pick"]
        self.assertEqual(len(picks), 1)
        self.assertEqual((picks[0][3], picks[0][4]), (8, 8))

    def test_other_hotspot_proximity_penalty_applies_near_other_hotspots(self) -> None:
        solver = self._new_solver_shell()
        solver.state = SimpleNamespace(width=12, height=12)
        solver.config = SimpleNamespace(
            setup_hotspots=[(0, 0), (10, 10)],
            order_other_hotspot_penalty=2.0,
            order_other_hotspot_penalty_radius=5,
        )

        near_other = solver._other_hotspot_proximity_penalty((9, 9), assigned_hotspot=(0, 0))
        far_from_other = solver._other_hotspot_proximity_penalty((3, 3), assigned_hotspot=(0, 0))
        self.assertGreater(near_other, 0.0)
        self.assertEqual(far_from_other, 0.0)

    def test_persistent_hotspots_only_for_initially_assigned_robots(self) -> None:
        solver = self._new_solver_shell()
        solver.robots = [
            SimpleNamespace(id=0, x=1, y=1, last_t=0),
            SimpleNamespace(id=1, x=8, y=8, last_t=0),
        ]
        solver._setup_robot_by_hotspot = {(0, 10): 0}

        mapping = solver._assign_persistent_robot_hotspots()
        self.assertEqual(mapping, {0: (0, 10)})
        self.assertEqual(getattr(solver.robots[0], "assigned_hotspot", None), (0, 10))
        self.assertFalse(hasattr(solver.robots[1], "assigned_hotspot"))

    def test_unassigned_robot_order_origin_uses_current_position(self) -> None:
        solver = self._new_solver_shell()
        solver._robot_hotspot_by_id = {}
        robot = SimpleNamespace(id=3, x=4, y=7, last_t=0)

        first = solver._robot_assigned_hotspot(robot)
        self.assertEqual(first, (4, 7))

        robot.x, robot.y = 9, 2
        second = solver._robot_assigned_hotspot(robot)
        self.assertEqual(second, (9, 2))

    def test_build_non_hotspot_forbidden_cells_includes_three_cell_band_and_caps(self) -> None:
        solver = self._new_solver_shell()
        solver.state = SimpleNamespace(width=30, height=20)
        hotspot = (10, 0)
        jobs = [
            SimpleNamespace(target_xy=(x, 0))
            for x in range(10, 20)
        ] + [
            SimpleNamespace(target_xy=(x, 2))
            for x in range(10, 20)
        ]
        solver._setup_jobs_by_hotspot = {hotspot: jobs}
        solver._nearest_edge_anchor = lambda cell: cell

        forbidden = solver._build_non_hotspot_forbidden_cells()
        self.assertIn((10, 0), forbidden)
        self.assertIn((15, 1), forbidden)
        self.assertIn((19, 2), forbidden)
        self.assertIn((9, 0), forbidden)   # left cap
        self.assertIn((20, 0), forbidden)  # right cap
        self.assertNotIn((15, 3), forbidden)  # allowed boundary just outside 3-cell band

    def test_safe_plan_path_blocks_forbidden_target_for_unassigned_robot(self) -> None:
        solver = self._new_solver_shell()
        solver.config = SimpleNamespace(path_step_limit=10, astar_slow_ms=9999, astar_print_slow=False, astar_log_blocked=False)
        solver._astar_calls = 0
        solver._astar_total_ms = 0.0
        solver._astar_max_ms = 0.0
        solver._astar_blocked_calls = 0
        solver._astar_slow_calls = 0
        solver._log = lambda *_args, **_kwargs: None
        solver._robot_hotspot_by_id = {}
        solver._non_hotspot_forbidden_cells = {(5, 5)}
        solver.planner = SimpleNamespace(plan_path=lambda *_args, **_kwargs: [(1, 5, 5)])

        robot = SimpleNamespace(id=1, x=0, y=0, last_t=0, docks={})
        path = solver._safe_plan_path(robot, 5, 5)
        self.assertEqual(path, [])

    def test_safe_plan_path_passes_forbidden_cells_to_planner_for_unassigned_robot(self) -> None:
        solver = self._new_solver_shell()
        solver.config = SimpleNamespace(path_step_limit=10, astar_slow_ms=9999, astar_print_slow=False, astar_log_blocked=False)
        solver._astar_calls = 0
        solver._astar_total_ms = 0.0
        solver._astar_max_ms = 0.0
        solver._astar_blocked_calls = 0
        solver._astar_slow_calls = 0
        solver._log = lambda *_args, **_kwargs: None
        solver._robot_hotspot_by_id = {}
        solver._non_hotspot_forbidden_cells = {(2, 2), (3, 3)}
        captured = {}

        def _plan_path(_robot, _tx, _ty, *, max_path_steps, blocked_cells):
            captured["blocked_cells"] = set(blocked_cells)
            return []

        solver.planner = SimpleNamespace(plan_path=_plan_path)
        robot = SimpleNamespace(id=7, x=0, y=0, last_t=0, docks={})
        solver._safe_plan_path(robot, 6, 6)
        self.assertEqual(captured.get("blocked_cells"), {(2, 2), (3, 3)})

    def test_finalize_dock_pallet_state_removes_static_pallet_from_scheduler(self) -> None:
        solver = self._new_solver_shell()
        solver.config = SimpleNamespace(max_time=50)
        solver.scheduler = SimpleNamespace(
            pallets={(4, 4): 7, (5, 5): 9},
            _rebuild_indexes=lambda: None,
        )
        solver.pallet_id_by_coord = {(4, 4): 11, (5, 5): 12}
        solver._persistently_docked_pallet_ids = set()
        captured = {}
        solver._record_pallet_move = lambda **kwargs: captured.update(kwargs)

        solver._finalize_dock_pallet_state(pallet_id=11, old_xy=(4, 4), dock_t=9)
        self.assertNotIn((4, 4), solver.scheduler.pallets)
        self.assertNotIn((4, 4), solver.pallet_id_by_coord)
        self.assertIn(11, solver._persistently_docked_pallet_ids)
        self.assertEqual(captured.get("dock_t"), 9)
        self.assertEqual(captured.get("undock_t"), 49)

    def test_candidate_robots_for_dock_prefers_robots_without_existing_docks(self) -> None:
        solver = self._new_solver_shell()
        solver.config = SimpleNamespace(max_robots_per_suggestion=3, path_step_limit=50)
        solver.robots = [
            SimpleNamespace(id=0, x=0, y=0, last_t=1, docks={(1, 0): 1}),
            SimpleNamespace(id=1, x=1, y=1, last_t=2, docks={}),
            SimpleNamespace(id=2, x=2, y=2, last_t=3, docks={}),
        ]
        solver._retired_robot_ids = set()
        solver._has_pending_setup_for_robot = lambda _rid: False
        from jr_walker.logic import DockSuggestion
        suggestion = DockSuggestion(sku=5, plan=[1, 2], gain=1.0, pallet_xy=(0, 0))

        ranked = solver._candidate_robots_for_suggestion(suggestion)
        self.assertEqual([r.id for r in ranked], [1, 2])


if __name__ == "__main__":
    unittest.main()
