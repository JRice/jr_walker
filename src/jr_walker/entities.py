from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


@dataclass
class NestConfig:
    """Describes a single nest's location and pallet layout.

    anchor: (x, y) of the near/hotspot corner of Line A — must be on a map edge.
    line_c_pallets: per-position SKU list for Line C (0 = gap).  Its length
                    determines the number of positions in both Line A and Line C.

    Line A SKUs are derived automatically: all integers 1..total that are *not*
    listed in line_c_pallets, sorted ascending (lowest = hotspot end).
    """
    anchor: Tuple[int, int]
    line_c_pallets: List[int]   # 0 = gap; non-zero = SKU at that position

    @property
    def n_positions(self) -> int:
        return len(self.line_c_pallets)

    @property
    def line_a_skus(self) -> List[int]:
        """Fills Line A with the missing SKUs from 1..total, lowest first."""
        line_c_set = {sku for sku in self.line_c_pallets if sku != 0}
        n, k = self.n_positions, len(line_c_set)
        universe = set(range(1, n + k + 1))
        return sorted(universe - line_c_set)

    @property
    def fulfill_near(self) -> Tuple[int, int]:
        """Fulfill point at the hotspot (anchor) end of Line A."""
        nx, ny = self.anchor
        return (nx - 1, ny)

    @property
    def fulfill_far(self) -> Tuple[int, int]:
        """Fulfill point at the far end of Line A."""
        nx, ny = self.anchor
        return (nx + self.n_positions, ny)


class JobKind(Enum):
    SETUP = auto()    # Moving a pallet into the nest
    ORDER = auto()    # Fulfilling orders on the conveyor belt
    WAITING = auto()  # Idle between jobs
    DONE = auto()     # All work complete


@dataclass
class Pallet:
    id: int
    x: int
    y: int
    sku: int


@dataclass
class Order:
    id: int
    items: Counter  # sku -> quantity required
    assigned_tick: Optional[int] = None
    fulfilled_tick: Optional[int] = None

    @property
    def is_fulfilled(self) -> bool:
        return self.fulfilled_tick is not None

    @property
    def total_picks(self) -> int:
        return sum(self.items.values())


@dataclass
class Robot:
    id: int
    nest_id: Optional[Tuple[int, int]]   # anchor (x, y) of the assigned nest
    x: int
    y: int
    last_tick: int = -1
    inventory: Counter = field(default_factory=Counter)
    job: JobKind = JobKind.WAITING
    # Maps (dx, dy) offset from robot to pallet_id; pallet sits at (robot_x+dx, robot_y+dy)
    docks: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def footprint_at(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Absolute (x, y) cells occupied by the robot and all docked pallets."""
        cells = [(x, y)]
        for dx, dy in self.docks:
            cells.append((x + dx, y + dy))
        return cells


@dataclass
class ActionEntry:
    tick: int
    robot_id: int
    action: str  # move | pick | dock | undock | fulfill
    x: int
    y: int

    def format_line(self) -> str:
        return f"{self.tick} {self.robot_id} {self.action} {self.x} {self.y}"
