import collections
from typing import List, Dict, Tuple

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class OrderOptimizer:
    def __init__(self, pallets_dict: Dict[Tuple[int, int], int]):
        """
        pallets_dict: The {(x,y): sku} dictionary from our WarehouseState
        """
        # We need a reverse lookup: {sku: [(x1,y1), (x2,y2)...]}
        self.sku_locations = collections.defaultdict(list)
        for (x, y), sku in pallets_dict.items():
            self.sku_locations[sku].append((x, y))

    def find_tightest_cluster(self, order_skus: List[int]) -> Tuple[Dict[int, Tuple[int, int]], int]:
        """
        Given a list of unique SKUs needed for an order, finds the tightest physical grouping 
        of pallets that satisfy the order.
        
        Returns:
            best_cluster: {sku: (target_x, target_y)} mapping
            bounding_box_score: The perimeter of the bounding box containing the cluster
        """
        # 1. Sort SKUs by rarity (fewest pallets in the warehouse first)
        # This prevents O(N^K) explosion by forcing the cluster to form around the most constrained variable.
        sorted_skus = sorted(order_skus, key=lambda s: len(self.sku_locations[s]))
        
        anchor_sku = sorted_skus[0]
        other_skus = sorted_skus[1:]
        
        best_cluster = None
        min_bounding_box = float('inf')
        
        # 2. Iterate through the pallets of the RAREST SKU
        for anchor_pos in self.sku_locations[anchor_sku]:
            current_cluster = {anchor_sku: anchor_pos}
            
            # 3. For every other SKU, greedily grab the closest pallet to the anchor
            for sku in other_skus:
                closest_pallet = min(self.sku_locations[sku], 
                                     key=lambda p: manhattan_distance(anchor_pos, p))
                current_cluster[sku] = closest_pallet
                
            # 4. Calculate the Bounding Box of this cluster
            xs = [pos[0] for pos in current_cluster.values()]
            ys = [pos[1] for pos in current_cluster.values()]
            
            # Perimeter of the bounding box (smaller = tighter cluster)
            bbox_perimeter = 2 * ((max(xs) - min(xs)) + (max(ys) - min(ys)))
            
            if bbox_perimeter < min_bounding_box:
                min_bounding_box = bbox_perimeter
                best_cluster = current_cluster
                
        return best_cluster, min_bounding_box

    def sort_orders_by_cluster_efficiency(self, orders: List[collections.Counter]):
        """
        Takes all 1000 orders and sorts them by how tightly clustered their required pallets are.
        Orders that can be picked in a 5x5 area will bubble to the top.
        """
        scored_orders = []
        for i, order in enumerate(orders):
            unique_skus = list(order.keys())
            _, bbox_score = self.find_tightest_cluster(unique_skus)
            scored_orders.append({
                "order_idx": i,
                "order": order,
                "cluster_score": bbox_score
            })
            
        # Sort by smallest bounding box
        scored_orders.sort(key=lambda x: x["cluster_score"])
        return scored_orders

if __name__ == "__main__":
    from pathlib import Path
    
    # 1. Parse the file specifically for pallets and orders
    filepath = "data/BIG_ORDER.txt"
    if not Path(filepath).exists():
        print(f"Error: Could not find {filepath}")
        exit(1)
        
    with open(filepath, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('#')]
    
    # Skip Robots
    num_robots = int(lines.pop(0))
    for _ in range(num_robots): 
        lines.pop(0)
    
    # Parse Pallets
    num_pallets = int(lines.pop(0))
    pallets_dict = {}
    for _ in range(num_pallets):
        x, y, sku = map(int, lines.pop(0).split())
        pallets_dict[(x, y)] = sku
        
    # Parse Orders
    num_orders = int(lines.pop(0))
    orders = []
    for _ in range(num_orders):
        skus = list(map(int, lines.pop(0).split()))
        orders.append(collections.Counter(skus))

    print(f"Initialized Optimizer with {len(pallets_dict)} pallets and {len(orders)} orders...\n")
    
    # 2. Run the Optimization
    optimizer = OrderOptimizer(pallets_dict)
    scored_orders = optimizer.sort_orders_by_cluster_efficiency(orders)
    
    # 3. Print the Results
    print("🏆 TOP 5 EASIEST ORDERS (Tightest Clusters - Do these first!)")
    print("-" * 75)
    for i in range(5):
        o = scored_orders[i]
        score = o['cluster_score']
        print(f"Rank {i+1:03} | Order ID: {o['order_idx']:>4} | Perimeter Score: {score:>3} | SKUs: {dict(o['order'])}")
        
    print("\n💀 BOTTOM 5 HARDEST ORDERS (Widest Clusters - Dock candidates!)")
    print("-" * 75)
    for i in range(1, 6):
        o = scored_orders[-i]
        score = o['cluster_score']
        print(f"Rank {1000-i+1:03} | Order ID: {o['order_idx']:>4} | Perimeter Score: {score:>3} | SKUs: {dict(o['order'])}")

def analyze_big_order(file_path="data/BIG_ORDER.txt"):
    if not Path(file_path).exists():
        print(f"Error: {file_path} not found. Make sure it's in the root of jr_walker.")
        return

    with open(file_path, "r") as f:
        content = f.read().splitlines()

    # Skip robots
    num_robots = int(content[0])
    cursor = num_robots + 1
    
    # Skip pallets
    num_pallets = int(content[cursor])
    cursor += num_pallets + 1
    
    # Analyze Orders
    num_orders = int(content[cursor])
    cursor += 1
    
    order_data = content[cursor : cursor + num_orders]
    all_skus = []
    for line in order_data:
        all_skus.extend(map(int, line.split()))

    counter = collections.Counter(all_skus)
    total_picks = len(all_skus)
    
    print("-" * 30)
    print(f"DISTRIBUTION ANALYSIS")
    print(f"Total Orders: {num_orders}")
    print(f"Total Individual Picks: {total_picks}")
    print(f"Unique SKUs: {len(counter)}")
    print("-" * 30)
    print("TOP 10 HIGH-RUNNERS (The 'Bucket Brigade' Candidates):")
    
    running_total = 0
    for i, (sku, count) in enumerate(counter.most_common(10), 1):
        percentage = (count / total_picks) * 100
        running_total += percentage
        print(f"{i}. SKU {sku: >3}: {count: >5} picks ({percentage:.1f}%)")
    
    print("-" * 30)
    print(f"The Top 10 SKUs account for {running_total:.1f}% of all warehouse movement.")
    print("-" * 30)
