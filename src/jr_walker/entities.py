from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


@dataclass
class NestConfig:
    """Describes a single nest's location and pallet layout.

    anchor: (x, y) of the near/hotspot corner of Line A — must be on a map edge.
    line_c_pallets: per-position SKU list for Line C (0 = gap = robot entrance).
    warehouse_skus: set of all SKU IDs that exist in this warehouse input.
                    Must be set before line_a_pallets is used (call set_warehouse_skus).
    """
    anchor: Tuple[int, int]
    line_c_pallets: List[int]   # 0 = gap; non-zero = SKU at that position
    warehouse_skus: FrozenSet[int] = field(default_factory=frozenset)

    @property
    def n_positions(self) -> int:
        return len(self.line_c_pallets)

    def set_warehouse_skus(self, skus: Set[int]) -> None:
        object.__setattr__(self, "warehouse_skus", frozenset(skus))

    @property
    def line_a_pallets(self) -> List[int]:
        """Per-position SKU list for Line A (0 = empty slot).

        All warehouse SKUs not in Line C are placed contiguously from position 0
        (hotspot end), sorted ascending.  Any remaining positions are left empty (0).
        Raises if the warehouse SKU set has not been provided.
        """
        if not self.warehouse_skus:
            raise RuntimeError(
                "NestConfig.warehouse_skus must be set before using line_a_pallets. "
                "Call set_warehouse_skus() after loading pallets."
            )
        line_c_set = {sku for sku in self.line_c_pallets if sku != 0}
        line_a_skus = sorted(self.warehouse_skus - line_c_set)
        n = self.n_positions
        if len(line_a_skus) > n:
            raise RuntimeError(
                f"Nest at {self.anchor}: {len(line_a_skus)} Line A SKUs needed "
                f"but only {n} positions available. Add more positions to line_c_pallets."
            )
        result = [0] * n
        for i, sku in enumerate(line_a_skus):
            result[i] = sku
        return result

    @property
    def line_a_skus(self) -> List[int]:
        """Sorted non-zero SKUs assigned to Line A."""
        return [sku for sku in self.line_a_pallets if sku != 0]

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
