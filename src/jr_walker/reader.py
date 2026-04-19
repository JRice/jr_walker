from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Tuple

from jr_walker.entities import Order, Pallet, Robot


def parse_worklist(path: str, stride: int = 1) -> Tuple[List[Robot], List[Pallet], List[Order]]:
    """Parse BIG_ORDER.txt and return (robots, pallets, orders).

    stride: read every Nth order (1 = all, 10 = every 10th).
    """
    lines = _non_blank_lines(Path(path).read_text(encoding="utf-8"))
    idx = 0

    # --- robots ---
    num_robots = int(lines[idx]); idx += 1
    robots: List[Robot] = []
    for rid in range(num_robots):
        x, y = map(int, lines[idx].split()); idx += 1
        robots.append(Robot(id=rid, nest_id=-1, x=x, y=y))

    # --- pallets ---
    num_pallets = int(lines[idx]); idx += 1
    pallets: List[Pallet] = []
    for pid in range(num_pallets):
        x, y, sku = map(int, lines[idx].split()); idx += 1
        pallets.append(Pallet(id=pid, x=x, y=y, sku=sku))

    # --- orders ---
    num_orders = int(lines[idx]); idx += 1
    orders: List[Order] = []
    oid = 0
    for raw_idx in range(num_orders):
        skus = list(map(int, lines[idx].split())); idx += 1
        if raw_idx % stride == 0:
            orders.append(Order(id=oid, items=Counter(skus)))
            oid += 1

    return robots, pallets, orders


def _non_blank_lines(text: str) -> List[str]:
    out = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("\ufeff")
        if stripped and not stripped.startswith("#"):
            out.append(stripped)
    return out
