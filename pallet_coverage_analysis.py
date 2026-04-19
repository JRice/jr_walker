
#!/usr/bin/env python3
"""
Analyze BIG_ORDER.txt and answer:

    "How many orders can be fulfilled with the smallest number of pallets available?"

Assumption:
- Pallets have infinite supply.
- Therefore, duplicate pallets of the same SKU are redundant for *feasibility*.
- One pallet of SKU X is enough to satisfy any number of copies of SKU X in any order.
- So this analysis reduces to choosing a subset of SKU TYPES, not individual duplicate pallets.

Output:
1. Best achievable order count for each pallet-budget k (where k means k distinct SKU types / one pallet per SKU).
2. Minimal pallet budget required to cover 25%, 50%, 75%, 90%, 95%, and 100% of orders.
3. The SKU set that achieves each best value.
4. Optional JSON output for downstream tooling.

This is exact, not heuristic: with only 20 SKU types, we can evaluate all 2^20 subsets efficiently
using a subset-sum / SOS dynamic programming transform.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def parse_big_order(path: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]], List[List[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    it = iter(lines)

    num_robots = int(next(it))
    robots = [tuple(map(int, next(it).split())) for _ in range(num_robots)]

    num_pallets = int(next(it))
    pallets = [tuple(map(int, next(it).split())) for _ in range(num_pallets)]

    num_orders = int(next(it))
    orders = [list(map(int, next(it).split())) for _ in range(num_orders)]

    return robots, pallets, orders


def compute_exact_cover_stats(pallets: List[Tuple[int, int, int]], orders: List[List[int]]) -> Dict:
    all_skus = sorted({sku for _, _, sku in pallets})
    sku_to_bit = {sku: i for i, sku in enumerate(all_skus)}
    n = len(all_skus)

    if n > 25:
        raise ValueError(f"Expected a small SKU universe; got {n}. Exact subset DP may be too large.")

    # Order feasibility depends only on distinct SKUs present in the order.
    order_mask_counts = Counter()
    for order in orders:
        mask = 0
        for sku in set(order):
            mask |= 1 << sku_to_bit[sku]
        order_mask_counts[mask] += 1

    size = 1 << n
    freq = [0] * size
    for mask, count in order_mask_counts.items():
        freq[mask] = count

    # cover[S] = number of orders whose required mask is a subset of S
    cover = freq[:]
    for i in range(n):
        bit = 1 << i
        for mask in range(size):
            if mask & bit:
                cover[mask] += cover[mask ^ bit]

    best_by_k = [0] * (n + 1)
    best_mask_by_k = [0] * (n + 1)
    for mask in range(size):
        k = mask.bit_count()
        c = cover[mask]
        if c > best_by_k[k]:
            best_by_k[k] = c
            best_mask_by_k[k] = mask

    def mask_to_skus(mask: int) -> List[int]:
        return [all_skus[i] for i in range(n) if (mask >> i) & 1]

    thresholds = [0.25, 0.50, 0.75, 0.90, 0.95, 1.00]
    threshold_results = []
    total_orders = len(orders)

    for frac in thresholds:
        target = int(total_orders * frac + 0.999999999)
        found = None
        for k, count in enumerate(best_by_k):
            if count >= target:
                found = {
                    "fraction": frac,
                    "target_orders": target,
                    "min_pallets": k,
                    "covered_orders": count,
                    "skus": mask_to_skus(best_mask_by_k[k]),
                }
                break
        threshold_results.append(found)

    best_rows = []
    for k in range(n + 1):
        best_rows.append({
            "pallets": k,
            "covered_orders": best_by_k[k],
            "fraction": best_by_k[k] / total_orders if total_orders else 0.0,
            "skus": mask_to_skus(best_mask_by_k[k]),
        })

    # Map each SKU type to a canonical pallet location (first occurrence), useful if
    # you want one representative physical pallet per SKU.
    canonical_pallet_for_sku = {}
    for x, y, sku in pallets:
        canonical_pallet_for_sku.setdefault(sku, (x, y))

    return {
        "num_orders": total_orders,
        "num_pallets": len(pallets),
        "num_distinct_skus": n,
        "best_by_pallet_budget": best_rows,
        "threshold_results": threshold_results,
        "canonical_pallet_for_sku": canonical_pallet_for_sku,
    }


def print_report(stats: Dict) -> None:
    print(f"Orders: {stats['num_orders']}")
    print(f"Physical pallets in file: {stats['num_pallets']}")
    print(f"Distinct SKU types: {stats['num_distinct_skus']}")
    print()
    print("Best achievable coverage by pallet budget")
    print("(Here, 'pallet budget' means one representative pallet per distinct SKU type.)")
    print()
    print(f"{'pallets':>7}  {'orders':>6}  {'fraction':>8}  {'delta':>5}  skus")
    print(f"{'-'*7}  {'-'*6}  {'-'*8}  {'-'*5}  {'-'*40}")
    last_fract = 0.0
    for row in stats["best_by_pallet_budget"]:
        delta = row['fraction'] - last_fract
        last_fract = row['fraction']
        skus = ",".join(map(str, row["skus"]))
        print(f"{row['pallets']:>7}  {row['covered_orders']:>6}  {row['fraction']:>7.1%}  {delta:>5%}  {skus}")

    print()
    print("Coverage thresholds")
    print()
    for row in stats["threshold_results"]:
        if row is None:
            continue
        skus = ",".join(map(str, row["skus"]))
        print(
            f"{row['fraction']:.0%} of orders: "
            f"{row['min_pallets']} pallets -> {row['covered_orders']} orders; "
            f"SKUs {{{skus}}}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to BIG_ORDER.txt")
    parser.add_argument("--json-out", help="Optional path to write JSON results")
    args = parser.parse_args()

    robots, pallets, orders = parse_big_order(args.input_file)
    stats = compute_exact_cover_stats(pallets, orders)
    print_report(stats)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print()
        print(f"Wrote JSON to {out_path}")


if __name__ == "__main__":
    main()
