from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


@dataclass
class NestConfig:
    """Describes a single nest's location and pallet layout.

    anchor: (x, y) of the near-end corner of Line A — must be on a map edge.
    The edge is inferred from the anchor: y==0 → north, y==height-1 → south,
    x==0 → west, x==width-1 → east.

    Shape fields (line_a_length, pallets_c, etc.) will be added when variable-
    length nests are implemented.  For now the layout is fixed at 10 pallets
    per row, two rows deep.
    """
    anchor: Tuple[int, int]


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
