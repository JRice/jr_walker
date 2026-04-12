import collections
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
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


def _in_bounds(x: int, y: int, width: int = 60, height: int = 40) -> bool:
    return 0 <= x < width and 0 <= y < height


def _parse_inferred_run_id(solution_path: Path) -> int | None:
    # Accepts stems like: solution_11582_1, test_solution_1428_2, partial_solution_100_7
    match = re.match(r"^(?:test_)?(?:partial_)?solution_\d+_(\d+)$", solution_path.stem)
    if not match:
        return None
    return int(match.group(1))


def _counter_to_sku_list(counter: collections.Counter) -> List[int]:
    out: List[int] = []
    for sku, count in sorted(counter.items()):
        out.extend([sku] * count)
    return out


@dataclass(slots=True)
class CellMetadata:
    use: int = 0
    collision_risk: int = 0
    picks: int = 0
    pick_travel_time_total: int = 0
    sku_map: collections.Counter = field(default_factory=collections.Counter)


@dataclass(slots=True)
class SolutionMetadata:
    width: int
    height: int
    cells: List[CellMetadata]
    fulfills: List[dict]
    robot_stats: List[dict]
    makespan: int
    ticks_analyzed: int

    def cell(self, x: int, y: int) -> CellMetadata:
        return self.cells[y * self.width + x]


def _build_solution_metadata(
    *,
    num_robots: int,
    robot_starts: List[Tuple[int, int]],
    pallet_defs: List[Tuple[int, int, int]],
    order_defs: List[collections.Counter],
    actions_by_timestep: Dict[int, List[Tuple[int, str, int, int]]],
    max_timestep: int,
):
    """
    Algorithm: single-pass time-step simulation over recorded actions.
    Pattern: phased tick pipeline matching validator semantics
    (undock -> pick -> dock -> move -> fulfill).
    """
    width = 60
    height = 40
    metadata = SolutionMetadata(
        width=width,
        height=height,
        cells=[CellMetadata() for _ in range(width * height)],
        fulfills=[],
        robot_stats=[],
        makespan=max_timestep,
        ticks_analyzed=max_timestep + 1 if max_timestep >= 0 else 0,
    )

    storage: List[collections.Counter] = [collections.Counter() for _ in range(num_robots)]
    robot_pos: List[List[int]] = [[x, y] for x, y in robot_starts]
    robot_docked: List[List[int]] = [[] for _ in range(num_robots)]
    robot_bundle_start: List[int | None] = [None for _ in range(num_robots)]
    robot_last_pick_or_fulfill_tick = [0 for _ in range(num_robots)]
    robot_idle_ticks = [0 for _ in range(num_robots)]
    robot_empty_moves = [0 for _ in range(num_robots)]
    robot_order_times: List[List[int]] = [[] for _ in range(num_robots)]
    local_orders: list[OrderState] = [
        OrderState(order_id=oid, items=Counter(items)) for oid, items in enumerate(order_defs)
    ]

    pallets: List[dict] = []
    pallet_at: Dict[int, int] = {}
    for pid, (x, y, sku) in enumerate(pallet_defs):
        pallets.append({"id": pid, "x": x, "y": y, "sku": sku, "docked_to": None})
        pallet_at[100 * y + x] = pid

    for t in range(max_timestep + 1):
        # Per-cell snapshot metrics at start of timestep t.
        for rid in range(num_robots):
            rx, ry = robot_pos[rid]
            if _in_bounds(rx, ry, width=width, height=height):
                metadata.cell(rx, ry).use += 1
                for sku, count in storage[rid].items():
                    metadata.cell(rx, ry).sku_map[sku] += count

        for pallet in pallets:
            px = pallet["x"]
            py = pallet["y"]
            if _in_bounds(px, py, width=width, height=height):
                metadata.cell(px, py).use += 1

        for rid in range(num_robots):
            rx, ry = robot_pos[rid]
            near_count = 0
            for other in range(num_robots):
                if other == rid:
                    continue
                ox, oy = robot_pos[other]
                if abs(rx - ox) + abs(ry - oy) <= 4:
                    near_count += 1
            if near_count > 0 and _in_bounds(rx, ry, width=width, height=height):
                metadata.cell(rx, ry).collision_risk += near_count

        actions = actions_by_timestep.get(t, [])
        acted_this_tick = set()
        for rid, _, _, _ in actions:
            acted_this_tick.add(rid)
        for rid in range(num_robots):
            if rid not in acted_this_tick:
                robot_idle_ticks[rid] += 1

        # Phase order matches validator: undock -> pick -> dock -> move -> fulfill
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
                continue
            sku = pallets[pid]["sku"]
            if not storage[rid]:
                robot_bundle_start[rid] = t
            storage[rid][sku] += 1

            if _in_bounds(x, y, width=width, height=height):
                travel_time = t - robot_last_pick_or_fulfill_tick[rid]
                cell = metadata.cell(x, y)
                cell.picks += 1
                cell.pick_travel_time_total += travel_time
            robot_last_pick_or_fulfill_tick[rid] = t

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
            robot_pos[rid][0] = x
            robot_pos[rid][1] = y
            if not storage[rid]:
                robot_empty_moves[rid] += 1

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

            bag = storage[rid]
            matched_order_id = -1
            for order in local_orders:
                if not order.fulfilled and order.items == bag:
                    order.fulfilled = True
                    matched_order_id = order.order_id
                    break

            sku_list = _counter_to_sku_list(storage[rid])
            metadata.fulfills.append(
                {
                    "x": x,
                    "y": y,
                    "skus": sku_list,
                    "robot": rid,
                    "timestep": t,
                    "order_id": matched_order_id,
                }
            )
            if robot_bundle_start[rid] is not None:
                robot_order_times[rid].append(t - robot_bundle_start[rid])
            robot_last_pick_or_fulfill_tick[rid] = t
            storage[rid].clear()
            robot_bundle_start[rid] = None

    for rid in range(num_robots):
        metadata.robot_stats.append(
            {
                "robot": rid,
                "idle": robot_idle_ticks[rid],
                "empty_moves": robot_empty_moves[rid],
                "order_times": robot_order_times[rid],
            }
        )

    return metadata


def _ensure_metadata_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata_runs (
            run_id INTEGER PRIMARY KEY,
            solution_path TEXT NOT NULL,
            worklist_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            makespan INTEGER NOT NULL,
            ticks_analyzed INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            num_robots INTEGER NOT NULL,
            num_pallets INTEGER NOT NULL,
            num_orders INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cell_metadata (
            run_id INTEGER NOT NULL,
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            use_score INTEGER NOT NULL,
            collision_risk INTEGER NOT NULL,
            picks INTEGER NOT NULL,
            pick_travel_time_total INTEGER NOT NULL,
            PRIMARY KEY (run_id, x, y),
            FOREIGN KEY (run_id) REFERENCES metadata_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS cell_sku_flow (
            run_id INTEGER NOT NULL,
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            sku INTEGER NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY (run_id, x, y, sku),
            FOREIGN KEY (run_id) REFERENCES metadata_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS fulfills (
            run_id INTEGER NOT NULL,
            fulfill_index INTEGER NOT NULL,
            timestep INTEGER NOT NULL,
            robot_id INTEGER NOT NULL,
            order_id INTEGER,
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            skus_json TEXT NOT NULL,
            PRIMARY KEY (run_id, fulfill_index),
            FOREIGN KEY (run_id) REFERENCES metadata_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS robot_stats (
            run_id INTEGER NOT NULL,
            robot_id INTEGER NOT NULL,
            idle_ticks INTEGER NOT NULL,
            empty_moves INTEGER NOT NULL,
            order_times_json TEXT NOT NULL,
            PRIMARY KEY (run_id, robot_id),
            FOREIGN KEY (run_id) REFERENCES metadata_runs(run_id) ON DELETE CASCADE
        );
        """
    )


def _choose_run_id(
    conn: sqlite3.Connection,
    *,
    requested_run_id: int | None,
    inferred_run_id: int | None,
) -> int:
    if requested_run_id is not None:
        return requested_run_id
    if inferred_run_id is not None:
        return inferred_run_id
    row = conn.execute("SELECT COALESCE(MAX(run_id), 0) FROM metadata_runs").fetchone()
    return int(row[0]) + 1


def store_solution_metadata(
    *,
    metadata: SolutionMetadata,
    db_path: Path,
    solution_path: Path,
    worklist_path: Path,
    num_robots: int,
    num_pallets: int,
    num_orders: int,
    run_id: int | None = None,
) -> int:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    solution_path = Path(solution_path)
    inferred_run_id = _parse_inferred_run_id(solution_path)

    conn = sqlite3.connect(db_path)
    try:
        _ensure_metadata_schema(conn)
        selected_run_id = _choose_run_id(
            conn,
            requested_run_id=run_id,
            inferred_run_id=inferred_run_id,
        )
        with conn:
            conn.execute("DELETE FROM metadata_runs WHERE run_id = ?", (selected_run_id,))
            conn.execute(
                """
                INSERT INTO metadata_runs (
                    run_id, solution_path, worklist_path, created_at,
                    makespan, ticks_analyzed, width, height,
                    num_robots, num_pallets, num_orders
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    selected_run_id,
                    str(solution_path),
                    str(worklist_path),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    metadata.makespan,
                    metadata.ticks_analyzed,
                    metadata.width,
                    metadata.height,
                    num_robots,
                    num_pallets,
                    num_orders,
                ),
            )

            cell_rows = []
            sku_rows = []
            for y in range(metadata.height):
                for x in range(metadata.width):
                    cell = metadata.cell(x, y)
                    cell_rows.append(
                        (
                            selected_run_id,
                            x,
                            y,
                            cell.use,
                            cell.collision_risk,
                            cell.picks,
                            cell.pick_travel_time_total,
                        )
                    )
                    top_skus = sorted(cell.sku_map.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
                    for sku, count in top_skus:
                        sku_rows.append((selected_run_id, x, y, sku, count))

            conn.executemany(
                """
                INSERT INTO cell_metadata (
                    run_id, x, y, use_score, collision_risk, picks, pick_travel_time_total
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                cell_rows,
            )
            if sku_rows:
                conn.executemany(
                    "INSERT INTO cell_sku_flow (run_id, x, y, sku, count) VALUES (?, ?, ?, ?, ?)",
                    sku_rows,
                )

            fulfill_rows = []
            for idx, fulfill in enumerate(metadata.fulfills):
                fulfill_rows.append(
                    (
                        selected_run_id,
                        idx,
                        int(fulfill.get("timestep", -1)),
                        int(fulfill["robot"]),
                        int(fulfill["x"]),
                        int(fulfill["y"]),
                        json.dumps(fulfill["skus"]),
                    )
                )
            if fulfill_rows:
                conn.executemany(
                    """
                    INSERT INTO fulfills (run_id, fulfill_index, timestep, robot_id, order_id, x, y, skus_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    fulfill_rows,
                )

            robot_rows = []
            for robot in metadata.robot_stats:
                robot_rows.append(
                    (
                        selected_run_id,
                        int(robot["robot"]),
                        int(robot["idle"]),
                        int(robot["empty_moves"]),
                        json.dumps(robot["order_times"]),
                    )
                )
            if robot_rows:
                conn.executemany(
                    """
                    INSERT INTO robot_stats (run_id, robot_id, idle_ticks, empty_moves, order_times_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    robot_rows,
                )
    finally:
        conn.close()
    return selected_run_id


def build_and_store_solution_metadata(
    solution_path: Path,
    worklist_path: Path = Path("docs/BIG_ORDER.txt"),
    metadata_db_path: Path = Path("output/solution_metadata.db"),
    metadata_run_id: int | None = None,
) -> int:
    solution_path = Path(solution_path)
    worklist_path = Path(worklist_path)
    if not solution_path.exists():
        raise FileNotFoundError(f"Solution file not found: {solution_path}")
    if not worklist_path.exists():
        raise FileNotFoundError(f"Worklist file not found: {worklist_path}")

    num_robots, robot_starts, pallet_defs, order_defs = _parse_worklist(worklist_path)
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

    metadata = _build_solution_metadata(
        num_robots=num_robots,
        robot_starts=robot_starts,
        pallet_defs=pallet_defs,
        order_defs=order_defs,
        actions_by_timestep=actions_by_timestep,
        max_timestep=max_timestep,
    )
    return store_solution_metadata(
        metadata=metadata,
        db_path=metadata_db_path,
        solution_path=solution_path,
        worklist_path=worklist_path,
        num_robots=num_robots,
        num_pallets=len(pallet_defs),
        num_orders=len(order_defs),
        run_id=metadata_run_id,
    )
