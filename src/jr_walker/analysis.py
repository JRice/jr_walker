import collections
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_worklist(worklist_path: Path):
    lines = [
        l.strip()
        for l in worklist_path.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")
    ]
    idx = 0
    num_robots = int(lines[idx])
    idx += 1
    idx += num_robots

    num_pallets = int(lines[idx])
    idx += 1
    pallet_sku: Dict[Tuple[int, int], int] = {}
    for _ in range(num_pallets):
        x, y, sku = map(int, lines[idx].split())
        idx += 1
        pallet_sku[(x, y)] = sku
    return num_robots, pallet_sku


def _edge_bucket(x: int, y: int, width: int = 60, height: int = 40) -> str:
    # Left/right edges each get their own bucket. Corners map to left/right.
    if x == 0:
        return "left_edge"
    if x == width - 1:
        return "right_edge"
    if y == 0:
        return "top_x0_29" if x < 30 else "top_x30_59"
    if y == height - 1:
        return "bottom_x0_29" if x < 30 else "bottom_x30_59"
    return "non_edge"


def analysis_output_path_for_solution(solution_path: Path, output_dir: Path = Path("output")) -> Path:
    solution_path = Path(solution_path)
    output_dir = Path(output_dir)
    stem = solution_path.stem
    if stem.startswith("solution_"):
        # solution_N.txt -> solution_N_analysis.txt
        analysis_name = f"{stem}_analysis.txt"
    else:
        analysis_name = f"{stem}_analysis.txt"
    return output_dir / analysis_name


def solution_analysis(
    solution_path: Path,
    worklist_path: Path = Path("data/BIG_ORDER.txt"),
    output_path: Path | None = None,
) -> Path:
    solution_path = Path(solution_path)
    worklist_path = Path(worklist_path)
    if not solution_path.exists():
        raise FileNotFoundError(f"Solution file not found: {solution_path}")
    if not worklist_path.exists():
        raise FileNotFoundError(f"Worklist file not found: {worklist_path}")

    num_robots, pallet_sku = _parse_worklist(worklist_path)
    storage: List[collections.Counter] = [collections.Counter() for _ in range(num_robots)]

    bucket_item_counts: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    bucket_fulfill_counts: collections.Counter = collections.Counter()
    bucket_total_item_counts: collections.Counter = collections.Counter()

    total_fulfills = 0
    total_items = 0
    unknown_pick_coords = 0

    for line in solution_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _t, rid_s, action, x_s, y_s = line.split()
        rid = int(rid_s)
        x = int(x_s)
        y = int(y_s)

        if rid < 0 or rid >= num_robots:
            continue

        if action == "pick":
            sku = pallet_sku.get((x, y))
            if sku is None:
                unknown_pick_coords += 1
                continue
            storage[rid][sku] += 1
            continue

        if action == "fulfill":
            bucket = _edge_bucket(x, y)
            if bucket == "non_edge":
                # Shouldn't happen for valid submissions, but still track if it does.
                bucket = "non_edge"
            bucket_fulfill_counts[bucket] += 1
            total_fulfills += 1
            for sku, cnt in storage[rid].items():
                bucket_item_counts[bucket][sku] += cnt
                bucket_total_item_counts[bucket] += cnt
                total_items += cnt
            storage[rid].clear()

    if output_path is None:
        output_path = analysis_output_path_for_solution(solution_path, output_dir=Path("output"))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ordered_buckets = [
        "left_edge",
        "right_edge",
        "top_x0_29",
        "top_x30_59",
        "bottom_x0_29",
        "bottom_x30_59",
        "non_edge",
    ]

    lines_out: List[str] = []
    lines_out.append(f"Solution Analysis: {solution_path}")
    lines_out.append(f"Worklist: {worklist_path}")
    lines_out.append("")
    lines_out.append(f"Total fulfills: {total_fulfills}")
    lines_out.append(f"Total items delivered: {total_items}")
    lines_out.append(f"Unknown pick coordinates: {unknown_pick_coords}")
    lines_out.append("")

    for bucket in ordered_buckets:
        fulfills = bucket_fulfill_counts[bucket]
        items = bucket_total_item_counts[bucket]
        lines_out.append(f"[{bucket}]")
        lines_out.append(f"  fulfills: {fulfills}")
        lines_out.append(f"  items: {items}")
        if items == 0:
            lines_out.append("  sku_counts: (none)")
            lines_out.append("")
            continue

        sku_counter = bucket_item_counts[bucket]
        sku_parts = [f"SKU{sku}:{count}" for sku, count in sorted(sku_counter.items())]
        lines_out.append("  sku_counts: " + ", ".join(sku_parts))
        lines_out.append("")

    output_path.write_text("\n".join(lines_out), encoding="utf-8")
    return output_path
