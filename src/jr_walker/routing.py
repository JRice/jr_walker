import heapq
from typing import List, Tuple
import numpy as np

def manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def find_path(robot, start_t: int, start_x: int, start_y: int, target_x: int, target_y: int, reservation_table: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Space-Time A* Pathfinding. Returns a list of (timestep, x, y) tuples.
    """
    open_set = []
    # Queue stores: (f_score, g_score, (time, x, y), path_so_far)
    heapq.heappush(open_set, (0, 0, (start_t, start_x, start_y), []))
    
    visited = set()
    visited.add((start_t, start_x, start_y))
    
    # 5-way movement: Up, Down, Left, Right, AND WAIT
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]
    
    while open_set:
        f_score, g_score, current_pos, path = heapq.heappop(open_set)
        ct, cx, cy = current_pos
        
        # Are we there yet?
        if cx == target_x and cy == target_y:
            return path
            
        # Time always moves forward
        nt = ct + 1 
        
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            
            # Have we checked this exact state in space AND time?
            if (nt, nx, ny) in visited:
                continue
                
            # Temporarily shift robot to test the move
            original_x, original_y = robot.x, robot.y
            robot.x, robot.y = cx, cy 
            
            can_fit = robot.can_move(dx, dy, nt, reservation_table)
            
            robot.x, robot.y = original_x, original_y # Reset
            
            if can_fit:
                visited.add((nt, nx, ny))
                new_path = path + [(nt, nx, ny)]
                
                # g_score is time elapsed. 
                # f_score = time elapsed + distance remaining
                new_g = g_score + 1
                new_f = new_g + manhattan((nx, ny), (target_x, target_y))
                
                heapq.heappush(open_set, (new_f, new_g, (nt, nx, ny), new_path))
                
    return [] # Blocked or ran out of time horizon!