import sqlite3
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from jr_walker.solver import RelocationJob, WarehouseSolver, PastRunAnalysis, _select_best_non_test_run_id  # noqa: E402
except Exception:  # pragma: no cover - environment-dependent optional import
    RelocationJob = None
    WarehouseSolver = None
    PastRunAnalysis = None
    _select_best_non_test_run_id = None


@unittest.skipIf(WarehouseSolver is None or RelocationJob is None, "solver dependencies unavailable")
class SolverMetadataPolicyTests(unittest.TestCase):
    def _workspace_case_dir(self) -> Path:
        base = ROOT / "output"
        base.mkdir(parents=True, exist_ok=True)
        case_dir = base / f"test_case_{uuid.uuid4().hex}"
        case_dir.mkdir(parents=True, exist_ok=False)
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
        return solver

    def test_periodic_dispatch_validation_runs_at_interval(self) -> None:
        solver = self._new_solver_shell()
        solver.config = SimpleNamespace(dispatch_validate_every_makespan=500)
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
        solver.config = SimpleNamespace(dispatch_validate_every_makespan=500)
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
                    (1, "output/test_solution_100.txt", 100),
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
                    (1, "output/test_solution_100.txt", 100),
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

    def test_build_travel_lane_cells_includes_metadata_high_use_halo(self) -> None:
        solver = self._new_solver_shell()
        solver.past_analysis.high_use_cells = {(15, 15)}

        lanes = solver._build_travel_lane_cells(lane_width=0)
        self.assertIn((15, 15), lanes)
        self.assertIn((15, 14), lanes)
        self.assertIn((16, 15), lanes)


if __name__ == "__main__":
    unittest.main()
