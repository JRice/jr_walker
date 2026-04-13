import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import _save_fulfill_rate_plot  # noqa: E402


class MainPlotTests(unittest.TestCase):
    def test_save_fulfill_rate_plot_writes_png_with_run_id(self) -> None:
        actions = [
            (0, 0, "move", 1, 0),
            (2, 0, "fulfill", 0, 0),
            (2, 1, "fulfill", 0, 0),
            (5, 0, "fulfill", 0, 0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = _save_fulfill_rate_plot(actions, 77, media_dir=Path(tmpdir))
            self.assertTrue(out.exists())
            self.assertEqual(out.name, "fulfill_rate_run_77.png")


if __name__ == "__main__":
    unittest.main()
