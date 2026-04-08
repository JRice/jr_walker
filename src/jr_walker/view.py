import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

class WarehouseState:
    def __init__(self, filepath="BIG_ORDER.txt", width=60, height=40):
        self.width = width
        self.height = height
        
        self.grid = np.zeros((self.height, self.width), dtype=int)
        
        # Paint the Fulfillment Zone (Perimeter = 1)
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
        self.robots = []
        self.pallets = {}
        self.orders = [] # <-- NEW: Orders are now natively part of the State
        
        self._parse_file(filepath)

    def _parse_file(self, filepath):
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Error: {filepath} not found.")

        with open(filepath, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('#')]
            
        # 1. Parse Robots
        num_robots = int(lines.pop(0))
        for _ in range(num_robots):
            x, y = map(int, lines.pop(0).split())
            self.robots.append((x, y))
            self.grid[y, x] = 3 
            
        # 2. Parse Pallets
        num_pallets = int(lines.pop(0))
        for _ in range(num_pallets):
            x, y, sku = map(int, lines.pop(0).split())
            self.pallets[(x, y)] = sku
            self.grid[y, x] = 2 
            
        # 3. Parse Orders
        num_orders = int(lines.pop(0))
        for _ in range(num_orders):
            skus = list(map(int, lines.pop(0).split()))
            self.orders.append(collections.Counter(skus))

    def visualize(self, path=None, output_path=None):
        """Renders the current tensor state with an optional path overlay and saves it."""
        # 0: White (Empty), 1: Light Green (Fulfillment), 2: Orange (Pallets), 3: Blue (Robots)
        cmap = ListedColormap(['#FFFFFF', '#D4EDDA', '#FD7E14', '#0D6EFD'])
        fig, ax = plt.subplots(figsize=(15, 10))
        
        ax.imshow(self.grid, cmap=cmap, origin='upper')
        
        # Grid lines and labels
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.2)
        ax.tick_params(which='major', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Path Overlay Logic
        if path:
            # Extract x and y, handling both (x, y) and (t, x, y) formats
            path_x = [p[1] if len(p)==3 else p[0] for p in path]
            path_y = [p[2] if len(p)==3 else p[1] for p in path]
            
            # Draw the path line
            ax.plot(path_x, path_y, color='red', linewidth=3, alpha=0.6, label='Planned Path')
            
            # Mark the start and end points
            ax.scatter(path_x[0], path_y[0], color='blue', s=100, zorder=5, label='Start')
            ax.scatter(path_x[-1], path_y[-1], color='red', s=100, marker='X', zorder=5, label='Target')

        plt.legend()
        plt.title("Warehouse Global Tensor State - Path Test")
        plt.tight_layout()
        if output_path is None:
            output_path = Path("media") / "warehouse_path.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
