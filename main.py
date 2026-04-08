import collections
from jr_walker.view import WarehouseState
from jr_walker.logic import OrderOptimizer
from jr_walker.entities import Robot
from jr_walker.routing import find_path, reserve_path
import numpy as np

def main():
    # 1. Initialize State & Visualizer
    # This also parses BIG_ORDER.txt internally
    state = WarehouseState("BIG_ORDER.txt")
    
    # 2. Analyze and Sort Orders
    optimizer = OrderOptimizer(state.pallets)
    # state.orders is populated during WarehouseState.__init__
    scored_orders = optimizer.sort_orders_by_cluster_efficiency(state.orders)
    
    # 3. Initialize the 3D Reservation Table
    # (Time, Y, X) - Start with 5000 timesteps
    max_time = 5000
    reservation_table = np.zeros((max_time, state.height, state.width))
    
    # Pre-fill the table with static obstacles (Pallets)
    # In a Space-Time A*, static obstacles exist at EVERY timestep
    for t in range(max_time):
        reservation_table[t] = (state.grid == 2).astype(int)

    # 4. Assign the first Robot to the easiest Order
    first_order = scored_orders[0]
    robot_0_start = state.robots[0] # (x, y)
    robot_0 = Robot(0, robot_0_start[0], robot_0_start[1])
    
    print(f"Routing Robot 0 to fulfill Order {first_order['order_idx']}...")
    
    # (This is where your logic loop would continue)
    # path = find_path(robot_0, 0, robot_0.x, robot_0.y, target_x, target_y, reservation_table)
    # if path:
    #     reserve_path(robot_0, path, reservation_table)

if __name__ == "__main__":
    main()