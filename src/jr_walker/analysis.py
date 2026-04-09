import collections
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_worklist(worklist_path: Path):
    lines = [
        l.strip().lstrip("\ufeff")
        for l in worklist_path.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    idx = 0
    num_robots = int(lines[idx])
    idx += 1

    robot_starts: List[Tuple[int, int]] = []
    for _ in range(num_robots):
        x, y = map(int, lines[idx].split())
        idx += 1
        robot_starts.append((x, y))

    num_pallets = int(lines[idx])
    idx += 1
    pallets: List[Tuple[int, int, int]] = []
    for _ in range(num_pallets):
        x, y, sku = map(int, lines[idx].split())
        idx += 1
        pallets.append((x, y, sku))

    num_orders = int(lines[idx])
    idx += 1
    orders: List[collections.Counter] = []
    for _ in range(num_orders):
        skus = list(map(int, lines[idx].split()))
        idx += 1
        orders.append(collections.Counter(skus))

    return num_robots, robot_starts, pallets, orders


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
    worklist_path: Path = Path("docs/BIG_ORDER.txt"),
    output_path: Path | None = None,
) -> Path:
    solution_path = Path(solution_path)
    worklist_path = Path(worklist_path)
    if not solution_path.exists():
        raise FileNotFoundError(f"Solution file not found: {solution_path}")
    if not worklist_path.exists():
        raise FileNotFoundError(f"Worklist file not found: {worklist_path}")

    num_robots, robot_starts, pallet_defs, order_defs = _parse_worklist(worklist_path)
    storage: List[collections.Counter] = [collections.Counter() for _ in range(num_robots)]
    robot_pos: List[List[int]] = [[x, y] for x, y in robot_starts]
    robot_docked: List[List[int]] = [[] for _ in range(num_robots)]
    robot_bundle_start: List[int | None] = [None for _ in range(num_robots)]

    pallets: List[dict] = []
    pallet_at: Dict[int, int] = {}
    for pid, (x, y, sku) in enumerate(pallet_defs):
        pallets.append({"id": pid, "x": x, "y": y, "sku": sku, "docked_to": None})
        pallet_at[100 * y + x] = pid

    orders_fulfilled = [False for _ in order_defs]
    order_fulfill_records: List[dict] = []

    bucket_item_counts: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    bucket_fulfill_counts: collections.Counter = collections.Counter()
    bucket_total_item_counts: collections.Counter = collections.Counter()
    robot_pick_counts: List[collections.Counter] = [collections.Counter() for _ in range(num_robots)]
    robot_skip_counts = [0 for _ in range(num_robots)]
    robot_distance = [0 for _ in range(num_robots)]
    robot_docked_ticks = [0 for _ in range(num_robots)]

    total_fulfills = 0
    total_items = 0
    unknown_pick_coords = 0

    actions_by_timestep: Dict[int, List[Tuple[int, str, int, int]]] = collections.defaultdict(list)
    max_timestep = -1

    for raw_line in solution_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        t_s, rid_s, action, x_s, y_s = parts[:5]
        t = int(t_s)
        rid = int(rid_s)
        action = action.lower()
        x = int(x_s)
        y = int(y_s)

        if rid < 0 or rid >= num_robots:
            continue

        actions_by_timestep[t].append((rid, action, x, y))
        if t > max_timestep:
            max_timestep = t

    for t in range(max_timestep + 1):
        actions = actions_by_timestep.get(t, [])

        acted_this_tick = set()
        for rid, _, _, _ in actions:
            acted_this_tick.add(rid)
        for rid in range(num_robots):
            if rid not in acted_this_tick:
                robot_skip_counts[rid] += 1
            if robot_docked[rid]:
                robot_docked_ticks[rid] += 1

        # Phase order matches submit validator: undock -> pick -> dock -> move -> fulfill
        for rid, action, x, y in actions:
            if action != "undock":
                continue
            pid = pallet_at.get(100 * y + x)
            if pid is None:
                continue
            pallet = pallets[pid]
            if pallet["docked_to"] != rid:
                continue
            pallet["docked_to"] = None
            robot_docked[rid] = [p for p in robot_docked[rid] if p != pid]

        for rid, action, x, y in actions:
            if action != "pick":
                continue
            pid = pallet_at.get(100 * y + x)
            if pid is None:
                unknown_pick_coords += 1
                continue
            sku = pallets[pid]["sku"]
            if not storage[rid]:
                robot_bundle_start[rid] = t
            storage[rid][sku] += 1
            robot_pick_counts[rid][sku] += 1

        for rid, action, x, y in actions:
            if action != "dock":
                continue
            pid = pallet_at.get(100 * y + x)
            if pid is None:
                continue
            pallet = pallets[pid]
            if pallet["docked_to"] is not None:
                continue
            pallet["docked_to"] = rid
            if pid not in robot_docked[rid]:
                robot_docked[rid].append(pid)

        for rid, action, x, y in actions:
            if action != "move":
                continue
            old_x, old_y = robot_pos[rid]
            dx = x - old_x
            dy = y - old_y
            robot_distance[rid] += abs(dx) + abs(dy)
            robot_pos[rid][0] = x
            robot_pos[rid][1] = y

            for pid in robot_docked[rid]:
                pallet = pallets[pid]
                old_key = 100 * pallet["y"] + pallet["x"]
                if old_key in pallet_at:
                    del pallet_at[old_key]
                pallet["x"] += dx
                pallet["y"] += dy
                pallet_at[100 * pallet["y"] + pallet["x"]] = pid

        for rid, action, x, y in actions:
            if action != "fulfill":
                continue

            bucket = _edge_bucket(x, y)
            bucket_fulfill_counts[bucket] += 1
            total_fulfills += 1
            for sku, cnt in storage[rid].items():
                bucket_item_counts[bucket][sku] += cnt
                bucket_total_item_counts[bucket] += cnt
                total_items += cnt

            matched_order_id = None
            if storage[rid]:
                for oid, order_bag in enumerate(order_defs):
                    if orders_fulfilled[oid]:
                        continue
                    if order_bag == storage[rid]:
                        orders_fulfilled[oid] = True
                        matched_order_id = oid
                        break

            if matched_order_id is not None:
                start_t = robot_bundle_start[rid] if robot_bundle_start[rid] is not None else t
                order_fulfill_records.append(
                    {
                        "order_id": matched_order_id,
                        "robot_id": rid,
                        "start_t": start_t,
                        "fulfill_t": t,
                        "duration": t - start_t,
                        "items": sum(order_defs[matched_order_id].values()),
                    }
                )

            storage[rid].clear()
            robot_bundle_start[rid] = None

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
    lines_out.append(f"Makespan (max timestep): {max_timestep}")
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

    lines_out.append("[robot_metrics]")
    for rid in range(num_robots):
        top_picks = robot_pick_counts[rid].most_common(10)
        if top_picks:
            top_picks_text = ", ".join([f"SKU{sku}:{cnt}" for sku, cnt in top_picks])
        else:
            top_picks_text = "(none)"
        lines_out.append(f"  robot_{rid}:")
        lines_out.append(f"    top_picked_skus: {top_picks_text}")
        lines_out.append(f"    skipped_ticks: {robot_skip_counts[rid]}")
        lines_out.append(f"    distance_traveled: {robot_distance[rid]}")
        lines_out.append(f"    docked_ticks: {robot_docked_ticks[rid]}")
    lines_out.append("")

    lines_out.append("[order_fulfillment_timing]")
    lines_out.append(f"  fulfilled_orders: {sum(1 for f in orders_fulfilled if f)} / {len(order_defs)}")

    most_time = sorted(
        order_fulfill_records,
        key=lambda r: (-r["duration"], -r["fulfill_t"], r["order_id"]),
    )[:100]
    least_time = sorted(
        order_fulfill_records,
        key=lambda r: (r["duration"], r["fulfill_t"], r["order_id"]),
    )[:100]

    lines_out.append("  Longest:")
    if not most_time:
        lines_out.append("    (none)")
    else:
        for rec in most_time:
            lines_out.append(
                "    "
                f"order_{rec['order_id']}: duration={rec['duration']} "
                f"(start={rec['start_t']}, fulfill={rec['fulfill_t']}), "
                f"robot={rec['robot_id']}, items={rec['items']}"
            )

    lines_out.append("  Shortest:")
    if not least_time:
        lines_out.append("    (none)")
    else:
        for rec in least_time:
            lines_out.append(
                "    "
                f"order_{rec['order_id']}: duration={rec['duration']} "
                f"(start={rec['start_t']}, fulfill={rec['fulfill_t']}), "
                f"robot={rec['robot_id']}, items={rec['items']}"
            )
    lines_out.append("")

    output_path.write_text("\n".join(lines_out), encoding="utf-8")
    return output_path
