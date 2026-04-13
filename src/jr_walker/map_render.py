from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def render_warehouse_map(
    *,
    width: int,
    height: int,
    pallet_items: list[tuple[tuple[int, int], int]],
    robot_cells: list[tuple[int, int, int]],
    title: str,
    output_path: Path,
) -> None:
    """Render a warehouse map image with perimeter, pallets (SKU labels), and robots."""
    grid = np.zeros((height, width), dtype=int)
    # 0: empty, 1: perimeter, 2: pallets, 3: robots
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    for (x, y), _sku in pallet_items:
        if 0 <= x < width and 0 <= y < height:
            grid[y, x] = 2

    for x, y, _rid in robot_cells:
        if 0 <= x < width and 0 <= y < height:
            grid[y, x] = 3

    cmap = ListedColormap(["#f8f7f4", "#d4f2d2", "#f7b267", "#7aa2f7"])
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.imshow(grid, cmap=cmap, origin="upper")

    if pallet_items:
        unique_skus = sorted({sku for _, sku in pallet_items})
        sku_to_idx = {sku: i for i, sku in enumerate(unique_skus)}
        palette = plt.cm.tab20(np.linspace(0.0, 1.0, 20))
        xs = [xy[0] for xy, _ in pallet_items]
        ys = [xy[1] for xy, _ in pallet_items]
        colors = [palette[sku_to_idx[sku] % 20] for _, sku in pallet_items]
        ax.scatter(xs, ys, c=colors, s=180, marker="s", edgecolors="black", linewidths=0.4, zorder=3)

        for (x, y), sku in pallet_items:
            ax.text(
                x,
                y,
                str(sku),
                ha="center",
                va="center",
                fontsize=6.8,
                color="black",
                zorder=4,
            )

    if robot_cells:
        for x, y, rid in robot_cells:
            ax.text(
                x,
                y,
                str(rid),
                ha="center",
                va="center",
                fontsize=8.0,
                color="white",
                zorder=5,
            )

    ax.set_title(title, fontsize=14)
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.35, alpha=0.15)
    ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
