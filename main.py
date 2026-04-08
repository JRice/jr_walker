import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Allow `python main.py` from repo root without installing the package.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jr_walker.entities import Robot
from jr_walker.logic import OrderOptimizer
from jr_walker.routing import find_path, reserve_path
from jr_walker.view import WarehouseState


def manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def choose_adjacent_pick_cell(
    state: WarehouseState, pallet_x: int, pallet_y: int
) -> Optional[Tuple[int, int]]:
    """Pick a free cell adjacent to a pallet for the robot to stand on."""
    candidates = [
        (pallet_x - 1, pallet_y),
        (pallet_x + 1, pallet_y),
        (pallet_x, pallet_y - 1),
        (pallet_x, pallet_y + 1),
    ]
    for x, y in candidates:
        in_bounds = 0 <= x < state.width and 0 <= y < state.height
        if not in_bounds:
            continue
        # Avoid other pallets and occupied robot spawn cells.
        if state.grid[y, x] in (2, 3):
            continue
        return x, y
    return None


def main():
    # 1. Initialize State & Visualizer
    state = WarehouseState("data/BIG_ORDER.txt")

    # 2. Analyze and Sort Orders
    optimizer = OrderOptimizer(state.pallets)
    scored_orders = optimizer.sort_orders_by_cluster_efficiency(state.orders)

    # 3. Initialize the 3D Reservation Table
    max_time = 500
    reservation_table = np.zeros((max_time, state.height, state.width), dtype=np.int8)

    # Pre-fill the table with static obstacles (Pallets)
    for t in range(max_time):
        # Value 2 marks static pallets as blocked in Robot.can_move (>1 is collision).
        reservation_table[t] = (state.grid == 2).astype(np.int8) * 2

    # 4. Pick the easiest order, then find the closest available robot to any item in it.
    first_order = scored_orders[0]
    robots = [Robot(idx, x, y) for idx, (x, y) in enumerate(state.robots)]

    candidates = []
    for sku in first_order["order"].keys():
        for pallet_x, pallet_y in optimizer.sku_locations[sku]:
            pick_cell = choose_adjacent_pick_cell(state, pallet_x, pallet_y)
            if pick_cell is None:
                continue
            target_x, target_y = pick_cell
            for robot in robots:
                distance = manhattan((robot.x, robot.y), (target_x, target_y))
                candidates.append((distance, robot.id, sku, pallet_x, pallet_y, target_x, target_y))

    if not candidates:
        print("No reachable pick cells were found for the easiest order.")
        return

    candidates.sort()
    print(f"Trying {len(candidates)} robot-target candidates for Order {first_order['order_idx']}...")

    chosen = None
    for _, robot_id, sku, pallet_x, pallet_y, target_x, target_y in candidates:
        robot = robots[robot_id]
        path = find_path(
            robot,
            0,
            robot.x,
            robot.y,
            target_x,
            target_y,
            reservation_table,
        )
        if path:
            chosen = (robot, sku, pallet_x, pallet_y, target_x, target_y, path)
            break

    if chosen is None:
        print("No path found for any robot to any item in the easiest order.")
        return

    robot, sku, pallet_x, pallet_y, target_x, target_y, path = chosen
    reserve_path(robot, path, reservation_table)

    output_path = ROOT / "media" / f"order_{first_order['order_idx']}_robot_{robot.id}.png"
    print(
        f"Selected Robot {robot.id} for SKU {sku} at pallet ({pallet_x}, {pallet_y}). "
        f"Path target is ({target_x}, {target_y}) with {len(path)} steps."
    )
    state.visualize(path=path, output_path=output_path)
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
