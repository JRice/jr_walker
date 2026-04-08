import sys
from pathlib import Path

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


def choose_adjacent_pick_cell(state: WarehouseState, pallet_x: int, pallet_y: int):
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

    # 4. Assign the first Robot to the easiest Order
    first_order = scored_orders[0]
    robot_0_start = state.robots[0]  # (x, y)
    robot_0 = Robot(0, robot_0_start[0], robot_0_start[1])

    target_cluster, _ = optimizer.find_tightest_cluster(list(first_order["order"].keys()))
    target_sku = next(iter(first_order["order"].keys()))
    pallet_x, pallet_y = target_cluster[target_sku]
    pick_cell = choose_adjacent_pick_cell(state, pallet_x, pallet_y)

    if pick_cell is None:
        print("No valid adjacent pick cell found for the chosen target pallet.")
        return

    target_x, target_y = pick_cell

    print(f"Routing Robot 0 to fulfill Order {first_order['order_idx']}...")

    path = find_path(
        robot_0,
        0,
        robot_0.x,
        robot_0.y,
        target_x,
        target_y,
        reservation_table,
    )

    if path:
        reserve_path(robot_0, path, reservation_table)
        print(f"Path found. {len(path)} steps.")
        state.visualize(path=path)
    else:
        print("No path found. Check if the target is blocked!")


if __name__ == "__main__":
    main()