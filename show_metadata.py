import argparse
import json
import sqlite3
import sys
import tomllib
from pathlib import Path

# Allow `python show_metadata.py` from repo root without installing the package.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jr_walker.map_render import render_warehouse_map
from jr_walker.view import WarehouseState


def _select_best_non_test_run_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        """
        SELECT run_id
        FROM metadata_runs
        WHERE solution_path NOT LIKE '%test_solution_%'
          AND solution_path NOT LIKE '%partial_solution_%'
        ORDER BY makespan ASC, run_id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is not None and row[0] is not None:
        return int(row[0])

    fallback = conn.execute(
        "SELECT run_id FROM metadata_runs ORDER BY makespan ASC, run_id DESC LIMIT 1"
    ).fetchone()
    if fallback is None or fallback[0] is None:
        return None
    return int(fallback[0])


def _load_paths_and_chunk_size(config_path: Path) -> tuple[Path, int, Path, Path]:
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    paths = data.get("paths", {})
    relocation = data.get("relocation", {})

    db_path = Path(paths.get("metadata_db_path", "output/solution_metadata.db"))
    input_path = Path(paths.get("input_path", "docs/BIG_ORDER.txt"))
    output_dir = Path(paths.get("output_dir", "output"))
    chunk_size_raw = relocation.get("relocate_chunk_size", 1)
    try:
        chunk_size = int(chunk_size_raw)
    except (TypeError, ValueError):
        chunk_size = 1
    chunk_size = max(1, chunk_size)
    return db_path, chunk_size, input_path, output_dir


def _render_tick0_sku_map(input_path: Path, output_path: Path) -> None:
    state = WarehouseState(str(input_path))
    render_warehouse_map(
        width=state.width,
        height=state.height,
        pallet_items=list(state.pallets.items()),
        robot_cells=[(x, y, rid) for rid, (x, y) in enumerate(state.robots)],
        title="Tick 0 SKU Map (Pallet cells labeled by SKU)",
        output_path=output_path,
    )


def _is_edge_cell(x: int, y: int, width: int, height: int) -> bool:
    return x == 0 or x == width - 1 or y == 0 or y == height - 1


def _chunk_edge_counts_with_skus(
    edge_fulfills: list[tuple[int, int, list[int]]],
    *,
    width: int,
    height: int,
    chunk_size: int,
) -> list[tuple[int, int, int, dict[int, int]]]:
    # chunk_key -> {"count": int, "wx": int, "wy": int, "sku": Counter}
    accum: dict[tuple[int, int], dict] = {}
    for x, y, skus in edge_fulfills:
        key = (x // chunk_size, y // chunk_size) if chunk_size > 1 else (x, y)
        cur = accum.get(key)
        if cur is None:
            cur = {"count": 0, "wx": 0, "wy": 0, "sku": {}}
        cur["count"] += 1
        cur["wx"] += x
        cur["wy"] += y
        sku_counter: dict[int, int] = cur["sku"]
        for sku in skus:
            sku_counter[sku] = sku_counter.get(sku, 0) + 1
        accum[key] = cur

    out: list[tuple[int, int, int, dict[int, int]]] = []
    for (chunk_x, chunk_y), data in accum.items():
        count_sum = int(data["count"])
        if count_sum <= 0:
            continue
        weighted_x = int(data["wx"])
        weighted_y = int(data["wy"])
        anchor_x = int(round(weighted_x / count_sum))
        anchor_y = int(round(weighted_y / count_sum))
        if chunk_size > 1:
            min_x = chunk_x * chunk_size
            min_y = chunk_y * chunk_size
            max_x = min(width - 1, min_x + chunk_size - 1)
            max_y = min(height - 1, min_y + chunk_size - 1)
            anchor_x = min(max(anchor_x, min_x), max_x)
            anchor_y = min(max(anchor_y, min_y), max_y)
        out.append((anchor_x, anchor_y, count_sum, dict(data["sku"])))

    out.sort(key=lambda row: (-row[2], row[1], row[0]))
    return out


def show_chunked_edge_fulfills(config_path: Path) -> int:
    db_path, chunk_size, input_path, output_dir = _load_paths_and_chunk_size(config_path)
    sku_map_path = output_dir / "metadata_tick0_sku_map.png"
    _render_tick0_sku_map(input_path=input_path, output_path=sku_map_path)
    print(f"Wrote tick-0 SKU map to {sku_map_path}")

    if not db_path.exists():
        print(f"Metadata DB not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    try:
        run_id = _select_best_non_test_run_id(conn)
        if run_id is None:
            print(f"No runs found in DB: {db_path}")
            return 1

        run_row = conn.execute(
            "SELECT width, height, solution_path, makespan FROM metadata_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if run_row is None:
            print(f"Run not found: run_id={run_id}")
            return 1
        width, height, solution_path, makespan = run_row
        width = int(width)
        height = int(height)

        raw_rows = conn.execute(
            """
            SELECT x, y, skus_json
            FROM fulfills
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
    finally:
        conn.close()

    edge_fulfills: list[tuple[int, int, list[int]]] = []
    for x, y, skus_json in raw_rows:
        xi = int(x)
        yi = int(y)
        if _is_edge_cell(xi, yi, width, height):
            try:
                raw_skus = json.loads(skus_json)
            except Exception:
                raw_skus = []
            skus: list[int] = []
            if isinstance(raw_skus, list):
                for sku_raw in raw_skus:
                    try:
                        skus.append(int(sku_raw))
                    except (TypeError, ValueError):
                        continue
            edge_fulfills.append((xi, yi, skus))

    chunked = _chunk_edge_counts_with_skus(
        edge_fulfills,
        width=width,
        height=height,
        chunk_size=chunk_size,
    )

    print("Best run edge-fulfill chunks")
    print(f"run_id={run_id} makespan={makespan} chunk_size={chunk_size}")
    print(f"solution_path={solution_path}")
    for x, y, count, sku_counts in chunked:
        print(f"{x},{y}: {count} fulfills")
        sku_rows = sorted(sku_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        for sku, delivered_count in sku_rows:
            print(f"- {delivered_count} x {sku}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show edge fulfill hotspots from best metadata run, chunked by relocate_chunk_size."
    )
    parser.add_argument(
        "--config",
        default="docs/config.toml",
        help="Path to config TOML (default: docs/config.toml).",
    )
    args = parser.parse_args()

    exit_code = show_chunked_edge_fulfills(Path(args.config))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
