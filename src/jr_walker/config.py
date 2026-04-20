from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import tomllib

from jr_walker.entities import NestConfig


@dataclass
class Config:
    # [warehouse]
    width: int
    height: int

    # [paths]
    input_path: str
    output_dir: str
    media_dir: str
    run_number_path: str

    # [limits]
    max_ticks: int
    max_runtime_minutes: float
    stride: int

    # [nests]
    nest_configs: List[NestConfig]

    # [pathfinding]
    strict_no_swap: bool
    max_idle_ticks: int
    stall_limit: int

    # [progress]
    order_interval: int
    tick_interval: int

    @property
    def nest_anchors(self) -> List[Tuple[int, int]]:
        return [nc.anchor for nc in self.nest_configs]


def load_config(path: str = "config/config.toml") -> Config:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))

    wh = raw.get("warehouse", {})
    paths = raw.get("paths", {})
    limits = raw.get("limits", {})
    pf = raw.get("pathfinding", {})
    prog = raw.get("progress", {})

    width = wh.get("width", 60)
    height = wh.get("height", 40)

    raw_nests = raw.get("nests", [])
    if not raw_nests:
        raw_nests = [
            {"coords": [15, 0], "line_c_pallets": [1,3,5,7,9,11,13,15,17,19]},
            {"coords": [35, 0], "line_c_pallets": [1,3,5,7,9,11,13,15,17,19]},
        ]
    nest_configs: List[NestConfig] = []
    for entry in raw_nests:
        x, y = int(entry["coords"][0]), int(entry["coords"][1])
        if not (x == 0 or x == width - 1 or y == 0 or y == height - 1):
            raise ValueError(
                f"Nest anchor ({x}, {y}) is not on a map edge "
                f"(warehouse is {width}×{height})."
            )
        line_c = [int(v) for v in entry.get("line_c_pallets", [1,3,5,7,9,11,13,15,17,19])]
        nest_configs.append(NestConfig(anchor=(x, y), line_c_pallets=line_c))

    return Config(
        width=width,
        height=height,
        input_path=paths.get("input_path", "config/BIG_ORDER.txt"),
        output_dir=paths.get("output_dir", "output"),
        media_dir=paths.get("media_dir", "media"),
        run_number_path=paths.get("run_number_path", "config/run_number.txt"),
        max_ticks=limits.get("max_ticks", 16000),
        max_runtime_minutes=limits.get("max_runtime_minutes", 240.0),
        stride=limits.get("stride", 1),
        nest_configs=nest_configs,
        strict_no_swap=pf.get("strict_no_swap", False),
        max_idle_ticks=pf.get("max_idle_ticks", 24),
        stall_limit=pf.get("stall_limit", 24),
        order_interval=prog.get("order_interval", 100),
        tick_interval=prog.get("tick_interval", 1000),
    )


def read_run_id(path: str) -> int:
    """Read, increment, and persist the run ID. Creates the file if missing."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = p.read_text(encoding="utf-8").strip()
        run_id = int(text) + 1 if text else 1
    except (FileNotFoundError, ValueError):
        run_id = 1
    p.write_text(str(run_id), encoding="utf-8")
    return run_id
