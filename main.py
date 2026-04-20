"""Entry point for the warehouse robot solver.

Run with:
    uv run python main.py
    uv run python main.py --config config/config.toml
"""
from __future__ import annotations

import atexit
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jr_walker.config import Config, load_config, read_run_id
from jr_walker.conveyor import plan_order_phase
from jr_walker.entities import ActionEntry, Order
from jr_walker.nest_builder import assign_robots_to_nests, plan_nest_construction
from jr_walker.output import last_tick_used, write_fulfillment_graph, write_solution
from jr_walker.reader import parse_worklist
from jr_walker.validator import SubmissionValidator, ValidationError
from jr_walker.warehouse import SpacetimeGrid


def main() -> None:
    cfg = load_config("config/config.toml")
    run_id = read_run_id(cfg.run_number_path)
    start_time = time.time()
    max_runtime_seconds = cfg.max_runtime_minutes * 60.0

    print(f"=== Run {run_id} ===")
    print(f"Input: {cfg.input_path}  stride={cfg.stride}")

    robots, pallets, orders = parse_worklist(cfg.input_path, stride=cfg.stride)
    print(f"Loaded: {len(robots)} robots, {len(pallets)} pallets, {len(orders)} orders")

    # --- Spacetime reservation grid ---
    grid = SpacetimeGrid(cfg.width, cfg.height, cfg.max_ticks)

    # Mark initial pallet positions as permanently occupied
    for p in pallets:
        grid.reserve_permanent(0, p.x, p.y)

    # Mark robot starting positions at t=0
    for r in robots:
        grid.reserve(0, r.x, r.y)

    # --- Assign robots to nests ---
    assign_robots_to_nests(robots, cfg.nest_configs)
    nest_robots = {
        nc.anchor: [r for r in robots if r.nest_id == nc.anchor]
        for nc in cfg.nest_configs
    }

    # Exclusion rectangles: each nest excludes the other nests' conveyor zones.
    def nest_rect(nc) -> tuple:
        nx, ny = nc.anchor
        return (nx - 2, ny, nx + nc.n_positions + 1, ny + 3)

    all_actions: List[ActionEntry] = []
    fulfilled_orders: List[Order] = []
    all_orders = list(orders)

    try:
        # ----------------------------------------------------------------
        # Phase 1: Nest construction (cooperative setup)
        # ----------------------------------------------------------------
        print("Phase 1: Building nests...")
        nest_finish_ticks = {}
        global_planned_ids: set = set()
        for nc in cfg.nest_configs:
            nr = nest_robots[nc.anchor]
            other_rects = [nest_rect(c) for c in cfg.nest_configs if c is not nc]
            finish_tick = plan_nest_construction(
                nest_config=nc,
                nest_robots=nr,
                pallets=pallets,
                grid=grid,
                actions=all_actions,
                other_nest_rects=other_rects,
                strict_no_swap=cfg.strict_no_swap,
                total_robot_count=len(robots),
                planned_ids=global_planned_ids,
            )
            nest_finish_ticks[nc.anchor] = finish_tick
            print(f"  Nest {nc.anchor} complete at tick {finish_tick} "
                  f"({len(nr)} robots assigned)")

        if time.time() - start_time > max_runtime_seconds:
            raise RuntimeError("Exceeded max_runtime after nest construction")

        # ----------------------------------------------------------------
        # Phase 2: Order fulfillment (conveyor belt)
        # ----------------------------------------------------------------
        print("Phase 2: Fulfilling orders...")

        all_pending = sorted(all_orders, key=lambda o: o.total_picks)
        nest_orders = {nc.anchor: [] for nc in cfg.nest_configs}
        for i, o in enumerate(all_pending):
            anchor = cfg.nest_configs[i % len(cfg.nest_configs)].anchor
            nest_orders[anchor].append(o)

        for nc in cfg.nest_configs:
            nr = nest_robots[nc.anchor]
            other_rects = [nest_rect(c) for c in cfg.nest_configs if c is not nc]
            new_fulfilled = plan_order_phase(
                nest_config=nc,
                nest_robots=nr,
                orders=nest_orders[nc.anchor],
                grid=grid,
                actions=all_actions,
                other_nest_rects=other_rects,
                strict_no_swap=cfg.strict_no_swap,
                stall_limit=cfg.stall_limit,
                max_idle_ticks=cfg.max_idle_ticks,
                max_ticks=cfg.max_ticks,
                start_time=start_time,
                max_runtime_seconds=max_runtime_seconds,
                progress_order_interval=cfg.order_interval,
                progress_tick_interval=cfg.tick_interval,
            )
            fulfilled_orders.extend(new_fulfilled)

    except (RuntimeError, KeyboardInterrupt) as exc:
        print(f"\n[!] Interrupted or errored: {exc}")
        traceback.print_exc()

    finally:
        _write_outputs(
            all_actions, fulfilled_orders, all_orders, cfg, run_id,
            start_time=start_time,
        )


def _write_outputs(
    actions: List[ActionEntry],
    fulfilled: List[Order],
    all_orders: List[Order],
    cfg: Config,
    run_id: int,
    start_time: float,
) -> None:
    max_tick = last_tick_used(actions)
    fulfilled_count = len(fulfilled)
    all_fulfilled = fulfilled_count == len(all_orders)

    solution_path = write_solution(actions, cfg.output_dir, run_id, max_tick, fulfilled_count)

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s — {fulfilled_count}/{len(all_orders)} orders fulfilled, "
          f"max_tick={max_tick}")

    # Run validator
    try:
        validator = SubmissionValidator(worklist_path=cfg.input_path)
        state = validator.validate_file(solution_path)
        print(f"Validation OK — {state.fulfilled_orders}/{state.total_orders} orders confirmed")
    except ValidationError as ve:
        print(f"Validation FAILED: {ve}")

    # Fulfillment rate graph (only when all orders filled)
    if all_fulfilled:
        write_fulfillment_graph(all_orders, cfg.media_dir, run_id, max_tick, fulfilled_count)


if __name__ == "__main__":
    main()
