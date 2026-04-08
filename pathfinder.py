import heapq
from typing import List, Tuple
import numpy as np

# Assuming we import Robot and WarehouseState from our simulator
# from simulator import Robot, WarehouseState

def manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def find_path(robot, start_x: int, start_y: int, target_x: int, target_y: int, global_grid: np.ndarray) -> List[Tuple[int, int]]:
    """
    Standard A* Pathfinding that uses the Robot's dynamic footprint for validation.
    """
    # Priority queue stores tuples of: (f_score, g_score, (current_x, current_y), path_so_far)
    open_set = []
    heapq.heappush(open_set, (0, 0, (start_x, start_y), []))
    
    # Keep track of visited nodes to prevent infinite loops
    visited = set()
    visited.add((start_x, start_y))
    
    # 4-way movement (Up, Down, Left, Right)
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    while open_set:
        f_score, g_score, current_pos, path = heapq.heappop(open_set)
        cx, cy = current_pos
        
        # Are we there yet?
        if cx == target_x and cy == target_y:
            return path
            
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            
            if (nx, ny) in visited:
                continue
                
            # THE MAGIC SAUCE: We ask the robot if it (and its docked pallets) can fit!
            # We temporarily shift the robot's coordinates to test the move.
            original_x, original_y = robot.x, robot.y
            robot.x, robot.y = cx, cy 
            can_fit = robot.can_move(dx, dy, global_grid)
            robot.x, robot.y = original_x, original_y # Reset
            
            if can_fit:
                visited.add((nx, ny))
                new_path = path + [(nx, ny)]
                new_g = g_score + 1
                new_f = new_g + manhattan((nx, ny), (target_x, target_y))
                
                heapq.heappush(open_set, (new_f, new_g, (nx, ny), new_path))
                
    return [] # No path found (Blocked!)