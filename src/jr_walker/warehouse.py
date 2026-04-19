from __future__ import annotations

from typing import List, Tuple

import numpy as np


class SpacetimeGrid:
    """3D spacetime reservation tensor.

    grid[t, y, x] == 1 means some entity (robot or pallet) occupies cell (x, y) at tick t.
    """

    def __init__(self, width: int, height: int, max_ticks: int) -> None:
        self.width = width
        self.height = height
        self.max_ticks = max_ticks
        self.grid: np.ndarray = np.zeros((max_ticks, height, width), dtype=np.int8)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, t: int, x: int, y: int) -> bool:
        if not self.in_bounds(x, y) or t < 0 or t >= self.max_ticks:
            return False
        return self.grid[t, y, x] == 0

    def reserve(self, t: int, x: int, y: int) -> None:
        if self.in_bounds(x, y) and 0 <= t < self.max_ticks:
            self.grid[t, y, x] = 1

    def reserve_permanent(self, t_from: int, x: int, y: int) -> None:
        """Mark (x, y) as occupied from t_from onward (e.g. a pallet placed here)."""
        if not self.in_bounds(x, y):
            return
        t = max(0, t_from)
        self.grid[t:, y, x] = 1

    def clear_from(self, t_from: int, x: int, y: int) -> None:
        """Remove occupancy from t_from onward (e.g. a pallet being picked up)."""
        if not self.in_bounds(x, y):
            return
        t = max(0, t_from)
        self.grid[t:, y, x] = 0

    def reserve_footprint(self, t: int, positions: List[Tuple[int, int]]) -> None:
        for x, y in positions:
            self.reserve(t, x, y)

    def is_footprint_free(self, t: int, positions: List[Tuple[int, int]]) -> bool:
        return all(self.is_free(t, x, y) for x, y in positions)

    def is_3x3_clear(self, cx: int, cy: int, t_start: int, ticks: int) -> bool:
        """True if every cell in the 3x3 patch centered on (cx, cy) is free for `ticks` ticks."""
        for dt in range(ticks):
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if not self.is_free(t_start + dt, cx + dx, cy + dy):
                        return False
        return True
