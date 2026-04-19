from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import tomllib


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
    nest_x_coords: List[int]

    # [pathfinding]
    strict_no_swap: bool
    max_idle_ticks: int
    stall_limit: int

    # [progress]
    order_interval: int
    tick_interval: int


def load_config(path: str = "config/config.toml") -> Config:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))

    wh = raw.get("warehouse", {})
    paths = raw.get("paths", {})
    limits = raw.get("limits", {})
    nests = raw.get("nests", {})
    pf = raw.get("pathfinding", {})
    prog = raw.get("progress", {})

    return Config(
        width=wh.get("width", 60),
        height=wh.get("height", 40),
        input_path=paths.get("input_path", "config/BIG_ORDER.txt"),
        output_dir=paths.get("output_dir", "output"),
        media_dir=paths.get("media_dir", "media"),
        run_number_path=paths.get("run_number_path", "config/run_number.txt"),
        max_ticks=limits.get("max_ticks", 16000),
        max_runtime_minutes=limits.get("max_runtime_minutes", 240.0),
        stride=limits.get("stride", 1),
        nest_x_coords=nests.get("x_coords", [15, 35]),
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
