import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import next_run_id  # noqa: E402


class RunNumberTests(unittest.TestCase):
    def test_next_run_id_creates_file_with_one_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run_number.txt"
            run_id = next_run_id(path)
            self.assertEqual(run_id, 1)
            self.assertEqual(path.read_text(encoding="utf-8").strip(), "1")

    def test_next_run_id_increments_existing_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run_number.txt"
            path.write_text("7\n", encoding="utf-8")
            run_id = next_run_id(path)
            self.assertEqual(run_id, 8)
            self.assertEqual(path.read_text(encoding="utf-8").strip(), "8")

    def test_next_run_id_resets_empty_file_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run_number.txt"
            path.write_text("", encoding="utf-8")
            run_id = next_run_id(path)
            self.assertEqual(run_id, 1)
            self.assertEqual(path.read_text(encoding="utf-8").strip(), "1")


if __name__ == "__main__":
    unittest.main()
