import collections
from typing import Dict, List, Tuple

import numpy as np


class Robot:
    def __init__(self, r_id: int, x: int, y: int):
        self.id = r_id
        self.x = x
        self.y = y
        self.storage = collections.Counter()  # For picking items
        
        # Maps a relative directional offset to a Pallet ID
        # e.g., {(0, -1): 42} means Pallet 42 is docked to the North
        self.docks: Dict[Tuple[int, int], int] = {} 
        
    def get_footprint(self, target_x: int, target_y: int) -> List[Tuple[int, int]]:
        """Returns the absolute coordinates this robot will occupy at a given target."""
        footprint = [(target_x, target_y)] # The robot's main body
        
        for (dx, dy) in self.docks.keys():
            footprint.append((target_x + dx, target_y + dy))
            
        return footprint

    def can_move(self, dx: int, dy: int, future_t: int, reservation_table: np.ndarray) -> bool:
        """Validates if the robot and attached pallets can safely exist at future_t."""
        target_x = self.x + dx
        target_y = self.y + dy
        
        # 1. Horizon Check: Have we exceeded our maximum allocated simulation time?
        max_t = reservation_table.shape[0]
        if future_t >= max_t:
            return False
            
        # 2. Get the footprint at the new location
        future_footprint = self.get_footprint(target_x, target_y)
        
        # 3. 3D Collision Detection
        for (fx, fy) in future_footprint:
            # Check grid boundaries
            if not (0 <= fy < reservation_table.shape[1] and 0 <= fx < reservation_table.shape[2]):
                return False
                
            # Check the tensor at the specific future timestep
            # We assume > 1 means a static pallet or another robot has claimed this space
            if reservation_table[future_t, fy, fx] > 1:
                return False 
                
        return True
