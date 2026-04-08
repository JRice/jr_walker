import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

class WarehouseState:
    def __init__(self, filepath="BIG_ORDER.txt", width=60, height=40):
        self.width = width
        self.height = height
        
        # Initialize the Tensor (y, x) format is standard for numpy matrices
        self.grid = np.zeros((self.height, self.width), dtype=int)
        
        # 1. Paint the Fulfillment Zone (Perimeter = 1)
        self.grid[0, :] = 1   # Top edge
        self.grid[-1, :] = 1  # Bottom edge
        self.grid[:, 0] = 1   # Left edge
        self.grid[:, -1] = 1  # Right edge
        
        self.robots = []
        self.pallets = {}
        
        self._parse_file(filepath)

    def _parse_file(self, filepath):
        if not Path(filepath).exists():
            print(f"Error: {filepath} not found.")
            return

        with open(filepath, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('#')]
            
        # Parse Robots
        num_robots = int(lines.pop(0))
        for i in range(num_robots):
            x, y = map(int, lines.pop(0).split())
            self.robots.append((x, y))
            self.grid[y, x] = 3 # Mark robot on tensor (Note: numpy is row, col -> y, x)
            
        # Parse Pallets
        num_pallets = int(lines.pop(0))
        for _ in range(num_pallets):
            x, y, sku = map(int, lines.pop(0).split())
            self.pallets[(x, y)] = sku
            self.grid[y, x] = 2 # Mark pallet on tensor

    def visualize(self):
        """Renders the current tensor state using matplotlib, with SKU X-Ray"""
        cmap = ListedColormap(['#FFFFFF', '#D4EDDA', '#FD7E14', '#0D6EFD'])
        fig, ax = plt.subplots(figsize=(18, 12)) # Slightly larger for text rendering
        
        cax = ax.imshow(self.grid, cmap=cmap, origin='upper')
        
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.2)
        ax.tick_params(which='major', bottom=False, left=False, labelbottom=False, labelleft=False)
        
        # --- NEW: TEXT ANNOTATIONS ---
        # Annotate the Pallets with their SKUs
        for (y, x), sku in self.pallets.items():
            # Highlight our high-runners (SKUs 1-4)
            if sku in [1, 2, 3, 4]:
                color = 'red'
                weight = 'bold'
                fontsize = 10
            else:
                color = 'black'
                weight = 'normal'
                fontsize = 6
                
            ax.text(x, y, str(sku), ha='center', va='center', 
                    color=color, fontweight=weight, fontsize=fontsize)
                    
        # Annotate the Robots with their IDs
        for i, (x, y) in enumerate(self.robots):
            ax.text(x, y, f"R{i}", ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=9)

        ax.set_title("Warehouse Global Tensor State (SKU X-Ray)", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    state = WarehouseState("data/BIG_ORDER.txt")
    state.visualize()
