import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jr_walker.analysis import (  # noqa: E402
    CellMetadata,
    SolutionMetadata,
    _build_solution_metadata,
    _parse_inferred_run_id,
    build_and_store_solution_metadata,
    store_solution_metadata,
)


class AnalysisMetadataTests(unittest.TestCase):
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

    def _single_cell_metadata(self) -> SolutionMetadata:
        return SolutionMetadata(
            width=1,
            height=1,
            cells=[CellMetadata()],
            fulfills=[],
            robot_stats=[],
            makespan=0,
            ticks_analyzed=1,
        )

    def test_parse_inferred_run_id(self) -> None:
        self.assertEqual(_parse_inferred_run_id(Path("solution_11582_1.txt")), 1)
        self.assertEqual(_parse_inferred_run_id(Path("test_solution_1428_2.txt")), 2)
        self.assertEqual(_parse_inferred_run_id(Path("partial_solution_999_7.txt")), 7)
        self.assertIsNone(_parse_inferred_run_id(Path("solution_11582.txt")))

    def test_build_solution_metadata_core_metrics(self) -> None:
        metadata = _build_solution_metadata(
            num_robots=1,
            robot_starts=[(1, 1)],
            pallet_defs=[(2, 1, 7)],
            actions_by_timestep={
                0: [(0, "pick", 2, 1)],
                1: [(0, "move", 0, 1)],
                2: [(0, "fulfill", 0, 1)],
            },
            max_timestep=2,
        )

        self.assertEqual(metadata.makespan, 2)
        self.assertEqual(metadata.ticks_analyzed, 3)
        self.assertEqual(metadata.cell(2, 1).use, 3)  # static pallet occupancy every tick
        self.assertEqual(metadata.cell(2, 1).picks, 1)
        self.assertEqual(metadata.cell(2, 1).pick_travel_time_total, 0)
        self.assertEqual(metadata.cell(0, 1).sku_map[7], 1)
        self.assertEqual(len(metadata.fulfills), 1)
        self.assertEqual(metadata.fulfills[0]["skus"], [7])
        self.assertEqual(metadata.fulfills[0]["robot"], 0)
        self.assertEqual(metadata.robot_stats[0]["idle"], 0)
        self.assertEqual(metadata.robot_stats[0]["empty_moves"], 0)
        self.assertEqual(metadata.robot_stats[0]["order_times"], [2])

    def test_store_solution_metadata_run_id_selection(self) -> None:
        tmp_path = self._workspace_case_dir()
        db_path = tmp_path / "meta.db"
        metadata = self._single_cell_metadata()

        run_id_1 = store_solution_metadata(
            metadata=metadata,
            db_path=db_path,
            solution_path=tmp_path / "solution_100_7.txt",
            worklist_path=tmp_path / "worklist.txt",
            num_robots=1,
            num_pallets=0,
            num_orders=0,
        )
        run_id_2 = store_solution_metadata(
            metadata=metadata,
            db_path=db_path,
            solution_path=tmp_path / "custom_solution_name.txt",
            worklist_path=tmp_path / "worklist.txt",
            num_robots=1,
            num_pallets=0,
            num_orders=0,
        )
        run_id_3 = store_solution_metadata(
            metadata=metadata,
            db_path=db_path,
            solution_path=tmp_path / "solution_200_8.txt",
            worklist_path=tmp_path / "worklist.txt",
            num_robots=1,
            num_pallets=0,
            num_orders=0,
            run_id=3,
        )

        self.assertEqual(run_id_1, 7)
        self.assertEqual(run_id_2, 8)
        self.assertEqual(run_id_3, 3)

    def test_store_solution_metadata_caps_sku_rows_to_top_20_per_cell(self) -> None:
        tmp_path = self._workspace_case_dir()
        db_path = tmp_path / "meta.db"
        cell = CellMetadata()
        for sku in range(30):
            cell.sku_map[sku] = sku + 1
        metadata = SolutionMetadata(
            width=1,
            height=1,
            cells=[cell],
            fulfills=[],
            robot_stats=[],
            makespan=0,
            ticks_analyzed=1,
        )
        run_id = store_solution_metadata(
            metadata=metadata,
            db_path=db_path,
            solution_path=tmp_path / "solution_1_1.txt",
            worklist_path=tmp_path / "worklist.txt",
            num_robots=1,
            num_pallets=0,
            num_orders=0,
        )
        conn = sqlite3.connect(db_path)
        try:
            row_count = conn.execute(
                "SELECT COUNT(*) FROM cell_sku_flow WHERE run_id = ? AND x = 0 AND y = 0",
                (run_id,),
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(row_count, 20)

    def test_build_and_store_solution_metadata_end_to_end(self) -> None:
        tmp_path = self._workspace_case_dir()
        worklist_path = tmp_path / "worklist.txt"
        solution_path = tmp_path / "solution_2_4.txt"
        db_path = tmp_path / "meta.db"

        worklist_path.write_text(
            "\n".join(
                [
                    "1",      # robots
                    "1 1",    # robot start
                    "1",      # pallets
                    "2 1 7",  # pallet
                    "1",      # orders
                    "7",      # one-item order
                ]
            ),
            encoding="utf-8",
        )
        solution_path.write_text(
            "\n".join(
                [
                    "0 0 pick 2 1",
                    "1 0 move 0 1",
                    "2 0 fulfill 0 1",
                ]
            ),
            encoding="utf-8",
        )

        run_id = build_and_store_solution_metadata(
            solution_path=solution_path,
            worklist_path=worklist_path,
            metadata_db_path=db_path,
        )
        self.assertEqual(run_id, 4)  # inferred from solution_2_4.txt

        conn = sqlite3.connect(db_path)
        try:
            fulfills = conn.execute(
                "SELECT COUNT(*) FROM fulfills WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            robots = conn.execute(
                "SELECT COUNT(*) FROM robot_stats WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(fulfills, 1)
        self.assertEqual(robots, 1)


if __name__ == "__main__":
    unittest.main()
