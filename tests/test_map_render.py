import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jr_walker.map_render import render_warehouse_map  # noqa: E402


class MapRenderTests(unittest.TestCase):
    def test_render_warehouse_map_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "warehouse_test.png"
            render_warehouse_map(
                width=8,
                height=6,
                pallet_items=[((1, 1), 3), ((2, 3), 7)],
                robot_cells=[(0, 0), (7, 5)],
                title="Unit Test Map",
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
