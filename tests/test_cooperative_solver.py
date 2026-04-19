import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jr_walker.cooperative_solver import CooperativeWarehouseSolver, NestState  # noqa: E402


class CooperativeSolverTests(unittest.TestCase):
    def test_nest_target_cells_rows_are_row0_then_row2(self) -> None:
        solver = CooperativeWarehouseSolver.__new__(CooperativeWarehouseSolver)
        cells = solver._nest_target_cells(15)
        self.assertEqual(len(cells), 20)
        self.assertEqual(cells[:10], [(15 + i, 0) for i in range(10)])
        self.assertEqual(cells[10:], [(15 + i, 2) for i in range(10)])

    def test_assign_robots_to_nests_gives_each_nest_at_least_one_robot(self) -> None:
        solver = CooperativeWarehouseSolver.__new__(CooperativeWarehouseSolver)
        solver.robots = [
            SimpleNamespace(id=0, x=2, y=5),
            SimpleNamespace(id=1, x=12, y=4),
            SimpleNamespace(id=2, x=40, y=4),
            SimpleNamespace(id=3, x=41, y=2),
            SimpleNamespace(id=4, x=25, y=3),
        ]
        nests = [
            NestState(nest_id=0, anchor=(15, 0)),
            NestState(nest_id=1, anchor=(35, 0)),
        ]
        solver._assign_robots_to_nests(nests)
        self.assertTrue(nests[0].robot_ids)
        self.assertTrue(nests[1].robot_ids)
        assigned = set(nests[0].robot_ids + nests[1].robot_ids)
        self.assertEqual(assigned, {0, 1, 2, 3, 4})


if __name__ == "__main__":
    unittest.main()
