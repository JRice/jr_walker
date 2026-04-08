import collections
from pathlib import Path

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

if __name__ == "__main__":
    analyze_big_order()
