import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jr_walker.validator import SubmissionValidator, ValidationError  # noqa: E402


class ValidatorMoveShapeConflictTests(unittest.TestCase):
    def test_docked_pallet_vs_moving_robot_target_is_rejected(self) -> None:
        worklist = "\n".join(
            [
                "2",
                "1 1",
                "3 2",
                "1",
                "2 1 7",
                "1",
                "7",
            ]
        )
        validator = SubmissionValidator(worklist_text=worklist)
        validator.validate_line("0 0 dock 2 1")
        with self.assertRaises(ValidationError) as ctx:
            validator.validate_line("1 0 move 1 2")
            validator.validate_line("1 1 move 2 2")
            validator.finalize()
        self.assertIn("error G2", str(ctx.exception))

    def test_two_docked_pallets_same_target_is_rejected(self) -> None:
        worklist = "\n".join(
            [
                "2",
                "1 1",
                "5 1",
                "2",
                "2 1 7",
                "4 1 8",
                "1",
                "7",
            ]
        )
        validator = SubmissionValidator(worklist_text=worklist)
        validator.validate_line("0 0 dock 2 1")
        validator.validate_line("0 1 dock 4 1")
        with self.assertRaises(ValidationError) as ctx:
            validator.validate_line("1 0 move 2 1")
            validator.validate_line("1 1 move 4 1")
            validator.finalize()
        self.assertIn("error H2", str(ctx.exception))

    def test_docked_pallet_swap_does_not_trigger_map_mismatch(self) -> None:
        worklist = "\n".join(
            [
                "2",
                "2 1",
                "4 2",
                "2",
                "3 1 7",
                "4 1 8",
                "1",
                "7",
            ]
        )
        validator = SubmissionValidator(worklist_text=worklist)
        validator.validate_line("0 0 dock 3 1")
        validator.validate_line("0 1 dock 4 1")
        # Robot 0's docked pallet moves into robot 1's docked pallet OLD cell (4,1),
        # which is legal because robot 1 moves and vacates it in the same tick.
        validator.validate_line("1 0 move 3 1")
        validator.validate_line("1 1 move 5 2")
        validator.validate_line("2 0 move 2 1")
        validator.finalize()


if __name__ == "__main__":
    unittest.main()
