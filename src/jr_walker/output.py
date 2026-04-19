from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from jr_walker.entities import ActionEntry, Order


def write_solution(
    actions: List[ActionEntry],
    output_dir: str,
    run_id: int,
    max_tick: int,
    fulfilled_count: int,
) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"run{run_id}_{max_tick}tick_{fulfilled_count}order_solution.txt"
    path = Path(output_dir) / filename
    sorted_actions = sorted(actions, key=lambda a: (a.tick, a.robot_id))
    lines = [a.format_line() for a in sorted_actions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Solution written to {path}")
    return path


def write_fulfillment_graph(
    orders: List[Order],
    media_dir: str,
    run_id: int,
    max_tick: int,
    fulfilled_count: int,
) -> Optional[Path]:
    """PNG: ticks on x-axis, cumulative fulfilled orders on y-axis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping graph")
        return None

    Path(media_dir).mkdir(parents=True, exist_ok=True)
    filename = f"run{run_id}_{max_tick}tick_{fulfilled_count}order_rate.png"
    path = Path(media_dir) / filename

    fulfilled_orders = sorted(
        [o for o in orders if o.is_fulfilled], key=lambda o: o.fulfilled_tick
    )
    ticks = [o.fulfilled_tick for o in fulfilled_orders]
    counts = list(range(1, len(fulfilled_orders) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ticks, counts, linewidth=1.5)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Fulfilled Orders")
    ax.set_title(f"Order fulfillment rate — run {run_id}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Graph written to {path}")
    return path


def last_tick_used(actions: List[ActionEntry]) -> int:
    return max((a.tick for a in actions), default=0)
