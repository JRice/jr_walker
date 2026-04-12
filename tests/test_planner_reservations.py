import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jr_walker.planner import ReservationPlanner  # noqa: E402


class PlannerReservationTests(unittest.TestCase):
    def test_can_add_static_obstacle_from_rejects_future_dynamic_conflict(self) -> None:
        planner = ReservationPlanner(
            static_blocked=np.zeros((4, 4), dtype=bool),
            width=4,
            height=4,
            max_time=8,
        )
        planner.reservation_table[6, 2, 1] = 3

        self.assertFalse(planner.can_add_static_obstacle_from(3, 1, 2))
        self.assertTrue(planner.can_add_static_obstacle_from(7, 1, 2))


if __name__ == "__main__":
    unittest.main()
