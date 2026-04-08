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

    def can_move(self, dx: int, dy: int, global_grid: np.ndarray) -> bool:
        """Validates if the robot and all attached pallets can move."""
        target_x = self.x + dx
        target_y = self.y + dy
        
        # 1. Get the footprint at the new location
        future_footprint = self.get_footprint(target_x, target_y)
        
        # 2. Check bounds and collisions
        for (fx, fy) in future_footprint:
            # Check grid boundaries
            if not (0 <= fy < global_grid.shape[0] and 0 <= fx < global_grid.shape[1]):
                return False
                
            # Check collisions on the tensor 
            # (Assuming 0 is empty, 1 is perimeter/walkable, 2 is pallet, 3 is robot)
            # NOTE: We must ignore our CURRENT footprint in this check!
            current_val = global_grid[fy, fx]
            if current_val > 1 and (fx, fy) not in self.get_footprint(self.x, self.y):
                return False 
                
        return True