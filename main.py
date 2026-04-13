import sys
from pathlib import Path
import argparse
import tomllib
import traceback
from dataclasses import dataclass, field
import re
import sqlite3
import time
import uuid

# Allow `python main.py` from repo root without installing the package.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jr_walker.validator import SubmissionValidator, ValidationError
from jr_walker.map_render import render_warehouse_map
from jr_walker.solver import SolverConfig, WarehouseSolver, PastRunAnalysis, load_best_past_analysis
from jr_walker.view import WarehouseState
from jr_walker.writer import write_actions
from jr_walker.analysis import build_and_store_solution_metadata


class UserInterruptError(ValidationError):
    """Raised when planning is interrupted by Ctrl-C after saving partial state."""


def _save_fulfill_rate_plot(
    actions: list[tuple[int, int, str, int, int]],
    run_id: int | None,
    *,
    media_dir: Path = Path("media"),
) -> Path:
    import matplotlib.pyplot as plt

    fulfills_by_timestep: dict[int, int] = {}
    for t, _rid, action, _x, _y in actions:
        if action != "fulfill":
            continue
        fulfills_by_timestep[int(t)] = fulfills_by_timestep.get(int(t), 0) + 1

    x_points: list[int] = [0]
    y_points: list[int] = [0]
    cumulative = 0
    for t in sorted(fulfills_by_timestep.keys()):
        cumulative += fulfills_by_timestep[t]
        x_points.append(int(t))
        y_points.append(int(cumulative))

    run_label = str(int(run_id)) if run_id is not None else "unknown"
    media_dir.mkdir(parents=True, exist_ok=True)
    output_path = media_dir / f"fulfill_rate_run_{run_label}.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(x_points, y_points, where="post", linewidth=2.0)
    ax.set_xlabel("Timestep (Makespan Axis)")
    ax.set_ylabel("Cumulative Fulfill Count")
    ax.set_title(f"Fulfill Rate Over Time (run_id={run_label})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _save_final_warehouse_map(
    actions: list[tuple[int, int, str, int, int]],
    *,
    run_id: int | None,
    worklist_path: str | Path,
    media_dir: Path = Path("media"),
) -> Path:
    validator = SubmissionValidator(worklist_path=worklist_path)
    for t, rid, action, x, y in actions:
        validator.validate_line(f"{t} {rid} {action} {x} {y}")
    final_state = validator.finalize()

    width = 60
    height = 40

    pallet_items: list[tuple[tuple[int, int], int]] = []
    for pallet in final_state.pallets:
        if pallet.docked_to_robot is not None:
            continue
        x = int(pallet.x)
        y = int(pallet.y)
        if 0 <= x < width and 0 <= y < height:
            pallet_items.append(((x, y), int(pallet.sku)))

    robot_cells: list[tuple[int, int]] = []
    for robot in final_state.robots:
        x = int(robot.x)
        y = int(robot.y)
        if 0 <= x < width and 0 <= y < height:
            robot_cells.append((x, y))

    run_label = str(int(run_id)) if run_id is not None else "unknown"
    output_path = media_dir / f"warehouse_final_run_{run_label}.png"
    render_warehouse_map(
        width=width,
        height=height,
        pallet_items=pallet_items,
        robot_cells=robot_cells,
        title=f"Final Tick Warehouse Map (run_id={run_label})",
        output_path=output_path,
    )
    return output_path


def make_unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 2
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def _parse_robot_key(raw_key: str) -> int:
    key = raw_key.strip().lower()
    if key.startswith("robot_"):
        key = key[len("robot_") :]
    if not key.isdigit():
        raise ValueError(f'Invalid robot key "{raw_key}" (expected e.g. "robot_0").')
    return int(key)


@dataclass
class RunConfig:
    max_runs: int = 1
    lane_width: int = 3
    relocation_edge_band: int = 6
    relocate_chunk_size: int = 1
    setup_hotspots: list[tuple[int, int]] = field(default_factory=list)
    setup_mini_box_radius: int = 2
    relocation_skus_to_relocate: list[int] | None = None
    max_time: int = 50000
    min_jobs_for_dock: int = 3
    max_makespan: int | None = None
    max_plan_time_seconds: float = 600.0
    relocation_top_skus: int = 8
    num_allowed_relocations: int = 10
    order_suggestion_gain_constant: float = 100.0
    dock_gain_scale: float = 2.0
    relocation_gain_scale: float = 1.5
    strict_no_swap: bool = False
    ticks_to_full_validation: int = 500
    astar_slow_ms: float = 40.0
    astar_print_slow: bool = False
    astar_log_blocked: bool = False
    enable_relocation_suggestions: bool = False
    suggestion_retry_limit: int = 12
    suggestion_backoff_base_cycles: int = 2
    suggestion_backoff_max_cycles: int = 128
    order_stagnation_cycle_limit: int = 256
    realistic_fail_mode: bool = False
    max_robots_per_suggestion: int = 3
    robot_fail_streak_for_parking: int = 3
    parking_candidate_limit: int = 96
    lns_enabled: bool = True
    lns_iterations: int = 60
    lns_window_actions: int = 28
    lns_tail_fraction: float = 0.35
    lns_max_shift: int = 2
    output_dir: str = "output"
    log_path: str = "output/run.log"
    metadata_db_path: str = "output/solution_metadata.db"
    input_path: str = "docs/BIG_ORDER.txt"
    every_n_orders: int = 1
    output_path: str | None = None
    metadata_run_id: int | None = None


def _get_table(data: dict, key: str) -> dict:
    section = data.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"If present, [{key}] must be a table.")
    return section


def _require_int(value, field_name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")
    return value


def _require_number(value, field_name: str, *, minimum: float | None = None) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    out = float(value)
    if minimum is not None and out < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")
    return out


def load_run_config(config_path: Path) -> RunConfig:
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    run_config = RunConfig()

    loop_section = _get_table(data, "loop")
    run_config.max_runs = _require_int(
        loop_section.get("max_runs", run_config.max_runs),
        "loop.max_runs",
        minimum=1,
    )

    paths_section = _get_table(data, "paths")
    run_config.output_dir = str(paths_section.get("output_dir", run_config.output_dir))
    run_config.log_path = str(paths_section.get("log_path", run_config.log_path))
    run_config.metadata_db_path = str(
        paths_section.get("metadata_db_path", run_config.metadata_db_path)
    )
    run_config.input_path = str(paths_section.get("input_path", run_config.input_path))
    raw_output_path = paths_section.get("output_path")
    if raw_output_path is not None:
        run_config.output_path = str(raw_output_path)

    relocation_section = _get_table(data, "relocation")
    run_config.lane_width = _require_int(
        relocation_section.get("lane_width", run_config.lane_width),
        "relocation.lane_width",
        minimum=0,
    )
    run_config.relocation_edge_band = _require_int(
        relocation_section.get("edge_band_for_heatmap", run_config.relocation_edge_band),
        "relocation.edge_band_for_heatmap",
        minimum=0,
    )
    run_config.relocate_chunk_size = _require_int(
        relocation_section.get("relocate_chunk_size", run_config.relocate_chunk_size),
        "relocation.relocate_chunk_size",
        minimum=1,
    )
    raw_setup_hotspots = relocation_section.get("setup_hotspots", run_config.setup_hotspots)
    if not isinstance(raw_setup_hotspots, list):
        raise ValueError("relocation.setup_hotspots must be a list of [x, y] pairs.")
    parsed_hotspots: list[tuple[int, int]] = []
    for idx, cell in enumerate(raw_setup_hotspots):
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            raise ValueError(f"relocation.setup_hotspots[{idx}] must be [x, y].")
        x = _require_int(cell[0], f"relocation.setup_hotspots[{idx}][0]", minimum=0)
        y = _require_int(cell[1], f"relocation.setup_hotspots[{idx}][1]", minimum=0)
        parsed_hotspots.append((x, y))
    run_config.setup_hotspots = parsed_hotspots

    raw_reloc_skus = relocation_section.get("skus_to_relocate")
    if raw_reloc_skus is not None:
        if not isinstance(raw_reloc_skus, list):
            raise ValueError("relocation.skus_to_relocate must be a list of integers.")
        run_config.relocation_skus_to_relocate = [
            _require_int(v, f"relocation.skus_to_relocate[{i}]", minimum=1)
            for i, v in enumerate(raw_reloc_skus)
        ]
    solver_section = _get_table(data, "solver")
    run_config.max_time = _require_int(
        solver_section.get("max_time", run_config.max_time),
        "solver.max_time",
        minimum=2,
    )
    run_config.min_jobs_for_dock = _require_int(
        solver_section.get("min_jobs_for_dock", run_config.min_jobs_for_dock),
        "solver.min_jobs_for_dock",
        minimum=1,
    )

    run_config.num_allowed_relocations = _require_int(
        solver_section.get("num_allowed_relocations", run_config.num_allowed_relocations),
        "solver.num_allowed_relocations",
        minimum=0,
    )
    run_config.relocation_top_skus = _require_int(
        solver_section.get("relocation_top_skus", run_config.relocation_top_skus),
        "solver.relocation_top_skus",
        minimum=1,
    )
    run_config.order_suggestion_gain_constant = _require_number(
        solver_section.get("order_suggestion_gain_constant", run_config.order_suggestion_gain_constant),
        "solver.order_suggestion_gain_constant",
        minimum=0.0,
    )
    run_config.dock_gain_scale = _require_number(
        solver_section.get("dock_gain_scale", run_config.dock_gain_scale),
        "solver.dock_gain_scale",
        minimum=0.0,
    )
    run_config.relocation_gain_scale = _require_number(
        solver_section.get("relocation_gain_scale", run_config.relocation_gain_scale),
        "solver.relocation_gain_scale",
        minimum=0.0,
    )
    raw_strict_no_swap = solver_section.get("strict_no_swap", run_config.strict_no_swap)
    if not isinstance(raw_strict_no_swap, bool):
        raise ValueError("solver.strict_no_swap must be a boolean.")
    run_config.strict_no_swap = raw_strict_no_swap
    run_config.astar_slow_ms = _require_number(
        solver_section.get("astar_slow_ms", run_config.astar_slow_ms),
        "solver.astar_slow_ms",
        minimum=0.0,
    )
    raw_astar_print_slow = solver_section.get("astar_print_slow", run_config.astar_print_slow)
    if not isinstance(raw_astar_print_slow, bool):
        raise ValueError("solver.astar_print_slow must be a boolean.")
    run_config.astar_print_slow = raw_astar_print_slow
    raw_astar_log_blocked = solver_section.get("astar_log_blocked", run_config.astar_log_blocked)
    if not isinstance(raw_astar_log_blocked, bool):
        raise ValueError("solver.astar_log_blocked must be a boolean.")
    run_config.astar_log_blocked = raw_astar_log_blocked
    raw_enable_relocation = solver_section.get(
        "enable_relocation_suggestions", run_config.enable_relocation_suggestions
    )
    if not isinstance(raw_enable_relocation, bool):
        raise ValueError("solver.enable_relocation_suggestions must be a boolean.")
    run_config.enable_relocation_suggestions = raw_enable_relocation
    run_config.ticks_to_full_validation = _require_int(
        solver_section.get("ticks_to_full_validation", run_config.ticks_to_full_validation),
        "solver.ticks_to_full_validation",
        minimum=0,
    )
    run_config.suggestion_retry_limit = _require_int(
        solver_section.get("suggestion_retry_limit", run_config.suggestion_retry_limit),
        "solver.suggestion_retry_limit",
        minimum=1,
    )
    run_config.suggestion_backoff_base_cycles = _require_int(
        solver_section.get("suggestion_backoff_base_cycles", run_config.suggestion_backoff_base_cycles),
        "solver.suggestion_backoff_base_cycles",
        minimum=1,
    )
    run_config.suggestion_backoff_max_cycles = _require_int(
        solver_section.get("suggestion_backoff_max_cycles", run_config.suggestion_backoff_max_cycles),
        "solver.suggestion_backoff_max_cycles",
        minimum=1,
    )
    if run_config.suggestion_backoff_max_cycles < run_config.suggestion_backoff_base_cycles:
        raise ValueError(
            "solver.suggestion_backoff_max_cycles must be >= solver.suggestion_backoff_base_cycles."
        )
    run_config.max_robots_per_suggestion = _require_int(
        solver_section.get("max_robots_per_suggestion", run_config.max_robots_per_suggestion),
        "solver.max_robots_per_suggestion",
        minimum=1,
    )
    run_config.order_stagnation_cycle_limit = _require_int(
        solver_section.get("order_stagnation_cycle_limit", run_config.order_stagnation_cycle_limit),
        "solver.order_stagnation_cycle_limit",
        minimum=1,
    )
    raw_realistic_fail_mode = solver_section.get("realistic_fail_mode", run_config.realistic_fail_mode)
    if not isinstance(raw_realistic_fail_mode, bool):
        raise ValueError("solver.realistic_fail_mode must be a boolean.")
    run_config.realistic_fail_mode = raw_realistic_fail_mode
    run_config.setup_mini_box_radius = _require_int(
        solver_section.get("setup_mini_box_radius", run_config.setup_mini_box_radius),
        "solver.setup_mini_box_radius",
        minimum=1,
    )
    run_config.robot_fail_streak_for_parking = _require_int(
        solver_section.get("robot_fail_streak_for_parking", run_config.robot_fail_streak_for_parking),
        "solver.robot_fail_streak_for_parking",
        minimum=1,
    )
    run_config.parking_candidate_limit = _require_int(
        solver_section.get("parking_candidate_limit", run_config.parking_candidate_limit),
        "solver.parking_candidate_limit",
        minimum=1,
    )

    limits_section = _get_table(data, "limits")
    raw_max_makespan = limits_section.get("max_makespan", run_config.max_makespan)
    if raw_max_makespan is not None:
        run_config.max_makespan = _require_int(raw_max_makespan, "limits.max_makespan", minimum=0)

    # `max_plan_time` is in seconds.
    run_config.max_plan_time_seconds = _require_number(
        limits_section.get("max_plan_time", run_config.max_plan_time_seconds),
        "limits.max_plan_time",
        minimum=1.0,
    )
    run_config.every_n_orders = _require_int(
        limits_section.get("every_n_orders", run_config.every_n_orders),
        "limits.every_n_orders",
        minimum=1,
    )

    lns_section = _get_table(data, "lns")
    raw_lns_enabled = lns_section.get("enabled", run_config.lns_enabled)
    if not isinstance(raw_lns_enabled, bool):
        raise ValueError("lns.enabled must be a boolean.")
    run_config.lns_enabled = raw_lns_enabled
    run_config.lns_iterations = _require_int(
        lns_section.get("iterations", run_config.lns_iterations),
        "lns.iterations",
        minimum=0,
    )
    run_config.lns_window_actions = _require_int(
        lns_section.get("window_actions", run_config.lns_window_actions),
        "lns.window_actions",
        minimum=1,
    )
    run_config.lns_tail_fraction = _require_number(
        lns_section.get("tail_fraction", run_config.lns_tail_fraction),
        "lns.tail_fraction",
        minimum=0.0,
    )
    if run_config.lns_tail_fraction > 1.0:
        raise ValueError("lns.tail_fraction must be <= 1.0.")
    run_config.lns_max_shift = _require_int(
        lns_section.get("max_shift", run_config.lns_max_shift),
        "lns.max_shift",
        minimum=1,
    )

    metadata_section = _get_table(data, "metadata")
    raw_run_id = metadata_section.get("run_id")
    if raw_run_id is not None:
        run_config.metadata_run_id = _require_int(raw_run_id, "metadata.run_id", minimum=1)

    return run_config


def _build_solver(state: WarehouseState, run_config: RunConfig, temp_output_path: Path, past_analysis: PastRunAnalysis) -> WarehouseSolver:
    return WarehouseSolver(
        state,
        SolverConfig(
            max_time=run_config.max_time,
            min_jobs_for_dock=run_config.min_jobs_for_dock,
            max_makespan=run_config.max_makespan,
            max_plan_time_seconds=run_config.max_plan_time_seconds,
            output_path=Path(run_config.output_path) if run_config.output_path else temp_output_path,
            progress_every=50,
            log_path=Path(run_config.log_path),
            lane_width=run_config.lane_width,
            relocation_edge_band=run_config.relocation_edge_band,
            relocate_chunk_size=run_config.relocate_chunk_size,
            setup_hotspots=list(run_config.setup_hotspots),
            relocation_top_skus=run_config.relocation_top_skus,
            num_allowed_relocations=run_config.num_allowed_relocations,
            order_suggestion_gain_constant=run_config.order_suggestion_gain_constant,
            dock_gain_scale=run_config.dock_gain_scale,
            relocation_gain_scale=run_config.relocation_gain_scale,
            strict_no_swap=run_config.strict_no_swap,
            dispatch_validate_every_makespan=run_config.ticks_to_full_validation,
            astar_slow_ms=run_config.astar_slow_ms,
            astar_print_slow=run_config.astar_print_slow,
            astar_log_blocked=run_config.astar_log_blocked,
            enable_relocation_suggestions=run_config.enable_relocation_suggestions,
            setup_mini_box_radius=run_config.setup_mini_box_radius,
            suggestion_retry_limit=run_config.suggestion_retry_limit,
            suggestion_backoff_base_cycles=run_config.suggestion_backoff_base_cycles,
            suggestion_backoff_max_cycles=run_config.suggestion_backoff_max_cycles,
            order_stagnation_cycle_limit=run_config.order_stagnation_cycle_limit,
            realistic_fail_mode=run_config.realistic_fail_mode,
            max_robots_per_suggestion=run_config.max_robots_per_suggestion,
            robot_fail_streak_for_parking=run_config.robot_fail_streak_for_parking,
            parking_candidate_limit=run_config.parking_candidate_limit,
            worklist_path=Path(run_config.input_path),
            lns_enabled=run_config.lns_enabled,
            lns_iterations=run_config.lns_iterations,
            lns_window_actions=run_config.lns_window_actions,
            lns_tail_fraction=run_config.lns_tail_fraction,
            lns_max_shift=run_config.lns_max_shift,
        ),
        past_analysis=past_analysis
    )


def _run_pipeline(
    solver: WarehouseSolver, run_config: RunConfig, output_dir: Path
) -> tuple[list[tuple[int, int, str, int, int]], Exception | None]:
    solve_error: Exception | None = None
    actions: list[tuple[int, int, str, int, int]] = []
    base_actions: list[tuple[int, int, str, int, int]] = []
    try:
        print("Building a fresh base solution guided by past analysis...")
        base_actions = solver.find_solution()
        actions = list(base_actions)

        print(f"Running LNS optimization on {len(actions)} base actions...")
        actions = solver.optimize_actions(actions)

        print("Validating generated actions...")
        validator = SubmissionValidator(worklist_path=run_config.input_path)
        for t, rid, action, x, y in actions:
            validator.validate_line(f"{t} {rid} {action} {x} {y}")
        final_state = validator.finalize()
        if final_state.fulfilled_orders != final_state.total_orders:
            raise ValidationError(
                f"Validation incomplete: fulfilled {final_state.fulfilled_orders}/{final_state.total_orders} orders."
            )
        print(f"Validation passed. Fulfilled {final_state.fulfilled_orders}/{final_state.total_orders} orders by timestep {final_state.next_timestep}.")

    except KeyboardInterrupt:
        solve_error = UserInterruptError(
            "Validation interrupted: user pressed Ctrl-C; writing partial solution."
        )
        print(f"Solver/optimizer failed: {type(solve_error).__name__}: {solve_error!r}")
        if base_actions:
            print("Reverting to pre-LNS baseline actions...")
            actions = list(base_actions)
        elif actions:
            print("Writing best-known actions from current attempt...")
        else:
            print("Writing partial output from actions planned so far...")
            actions = solver.actions.sorted_actions()
        try:
            actions = solver._repair_idle_wait_conflicts(actions)
        except Exception:
            # Best-effort only; keep raw planned actions if repair fails.
            pass
    except Exception as exc:
        solve_error = exc
        print(f"Solver/optimizer failed: {type(exc).__name__}: {exc!r}")
        print(traceback.format_exc())
        if base_actions:
            print("Reverting to pre-LNS baseline actions...")
            actions = list(base_actions)
        elif actions:
            print("Writing best-known actions from current attempt...")
        else:
            print("Writing partial output from actions planned so far...")
            actions = solver.actions.sorted_actions()
        try:
            actions = solver._repair_idle_wait_conflicts(actions)
        except Exception:
            # Best-effort only; keep raw planned actions if repair fails.
            pass
    return actions, solve_error


def _save_and_report(
    actions: list[tuple[int, int, str, int, int]],
    state: WarehouseState,
    run_config: RunConfig,
    output_dir: Path,
    output_prefix: str,
    temp_output_path: Path,
    solve_error: Exception | None,
) -> None:
    makespan = max((t for t, _, _, _, _ in actions), default=-1)
    move_count = sum(1 for _, _, action, _, _ in actions if action == "move")

    final_output_path: Path
    metadata_run_id = run_config.metadata_run_id
    metadata_db_path = Path(run_config.metadata_db_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_config.output_path:
        final_output_path = Path(run_config.output_path)
        write_actions(actions, final_output_path)
        # Still attempt to store metadata even with an explicit output path.
        try:
            metadata_run_id = build_and_store_solution_metadata(
                solution_path=final_output_path,
                metadata_db_path=metadata_db_path,
                metadata_run_id=run_config.metadata_run_id,
            )
        except Exception as analysis_exc:
            print(f"Metadata persistence failed: {analysis_exc}")
    else:
        # Use a temporary placeholder path to break the circular dependency between
        # needing a run_id for the filename and needing a file to generate the run_id.
        placeholder_path = output_dir / f"temp_{uuid.uuid4().hex}.txt"
        write_actions(actions, placeholder_path)

        # Store metadata to get the run ID.
        try:
            metadata_run_id = build_and_store_solution_metadata(
                solution_path=placeholder_path,
                metadata_db_path=metadata_db_path,
                metadata_run_id=run_config.metadata_run_id,
            )
        except Exception as analysis_exc:
            print(f"Metadata persistence failed: {analysis_exc}")
            metadata_run_id = None

        # Now, construct the final path.
        name_prefix = f"{output_prefix}solution"
        if solve_error is not None:
            name_prefix = f"{output_prefix}partial_solution"

        if metadata_run_id is not None:
            final_output_path = output_dir / f"{name_prefix}_{makespan}_{metadata_run_id}.txt"
        else:
            # Fallback to old unique-suffix naming if metadata failed.
            final_output_path = make_unique_path(output_dir / f"{name_prefix}_{makespan}.txt")

        # Rename the placeholder to the final path.
        if placeholder_path.exists():
            if final_output_path.exists():
                final_output_path.unlink()
            placeholder_path.rename(final_output_path)

        # If we got a run ID, we need to update the path in the database.
        if metadata_run_id is not None:
            try:
                conn = sqlite3.connect(metadata_db_path)
                conn.execute(
                    "UPDATE metadata_runs SET solution_path = ? WHERE run_id = ?",
                    (str(final_output_path), metadata_run_id),
                )
                conn.commit()
            except sqlite3.Error as db_exc:
                print(f"Failed to update final solution path in DB: {db_exc}")
            finally:
                if "conn" in locals() and conn:
                    conn.close()

    if not run_config.output_path and temp_output_path.exists():
        temp_output_path.unlink()

    print(f"Wrote {len(actions)} actions to {final_output_path}")
    print(f"Move count: {move_count}")
    print(f"Plan makespan: {makespan} timesteps")
    print(f"Orders used: {len(state.orders)}")
    if metadata_run_id is not None:
        print(f"Wrote metadata to {metadata_db_path} (run_id={metadata_run_id})")

    plot_path = _save_fulfill_rate_plot(actions, metadata_run_id)
    print(f"Wrote fulfill-rate plot to {plot_path}")
    try:
        map_path = _save_final_warehouse_map(
            actions,
            run_id=metadata_run_id,
            worklist_path=run_config.input_path,
        )
        print(f"Wrote final warehouse map to {map_path}")
    except Exception as map_exc:
        print(f"Final warehouse map generation failed: {map_exc}")


def main():
    started_at = time.perf_counter()
    parser = argparse.ArgumentParser(description="Build a warehouse action plan.")
    parser.add_argument(
        "--config",
        default="docs/config.toml",
        help="TOML run config with solver and path parameters. Default: docs/config.toml",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    run_config = RunConfig()
    if config_path.exists():
        try:
            run_config = load_run_config(config_path)
        except Exception as exc:
            print(f"Failed to parse config {config_path}: {exc}")
            raise SystemExit(1)
    else:
        print(f"Config file not found at {config_path}; using defaults.")

    ## Main logic:

    state = WarehouseState(run_config.input_path)
    if run_config.every_n_orders > 1:
        state.orders = state.orders[:: run_config.every_n_orders]

    if config_path.exists():
        print(
            "Loaded config from "
            f"{config_path} "
            f"(max_runs={run_config.max_runs}, "
            f"lane_width={run_config.lane_width}, "
            f"edge_band_for_heatmap={run_config.relocation_edge_band}, "
            f"relocate_chunk_size={run_config.relocate_chunk_size}, "
            f"setup_hotspots={run_config.setup_hotspots}, "
            f"max_time={run_config.max_time}, "
            f"max_makespan={run_config.max_makespan}, "
            f"max_plan_time={run_config.max_plan_time_seconds:.1f}s, "
            f"relocation_top_skus={run_config.relocation_top_skus}, "
            f"num_allowed_relocations={run_config.num_allowed_relocations}, "
            f"dock_gain_scale={run_config.dock_gain_scale}, "
            f"relocation_gain_scale={run_config.relocation_gain_scale}, "
            f"strict_no_swap={run_config.strict_no_swap}, "
            f"ticks_to_full_validation={run_config.ticks_to_full_validation}, "
            f"astar_slow_ms={run_config.astar_slow_ms}, "
            f"enable_relocation_suggestions={run_config.enable_relocation_suggestions}, "
            f"suggestion_retry_limit={run_config.suggestion_retry_limit}, "
            f"suggestion_backoff_base_cycles={run_config.suggestion_backoff_base_cycles}, "
            f"suggestion_backoff_max_cycles={run_config.suggestion_backoff_max_cycles}, "
            f"order_stagnation_cycle_limit={run_config.order_stagnation_cycle_limit}, "
            f"realistic_fail_mode={run_config.realistic_fail_mode}, "
            f"max_robots_per_suggestion={run_config.max_robots_per_suggestion}, "
            f"setup_mini_box_radius={run_config.setup_mini_box_radius}, "
            f"robot_fail_streak_for_parking={run_config.robot_fail_streak_for_parking}, "
            f"parking_candidate_limit={run_config.parking_candidate_limit}, "
            f"forced_reloc_skus={run_config.relocation_skus_to_relocate})."
        )

    output_dir = Path(run_config.output_dir)
    output_prefix = f"stride_{run_config.every_n_orders}_" if run_config.every_n_orders > 1 else ""
    temp_output_path = output_dir / f"{output_prefix}solution_latest.txt"

    db_path = Path(run_config.metadata_db_path)
    any_success = False
    last_error: Exception | None = None

    for iteration in range(1, run_config.max_runs + 1):
        print(f"\n=== Iteration {iteration}/{run_config.max_runs} ===")
        past_analysis = PastRunAnalysis()
        if db_path.exists():
            print("Loading analysis of the best existing solution from SQL database...")
            past_analysis = load_best_past_analysis(db_path, state.width, state.height, state.pallets)
            if past_analysis.run_id >= 0:
                print(
                    f"Loaded analysis for Run #{past_analysis.run_id}: "
                    f"found {len(past_analysis.high_use_cells)} high-traffic cells to avoid."
                )

        print("Generating new Suggestions based on this analysis...")
        solver = _build_solver(state, run_config, temp_output_path, past_analysis)
        actions, solve_error = _run_pipeline(solver, run_config, output_dir)
        _save_and_report(
            actions,
            state,
            run_config,
            output_dir,
            output_prefix,
            temp_output_path,
            solve_error,
        )
        if solve_error is None:
            any_success = True
        else:
            last_error = solve_error
            if isinstance(solve_error, UserInterruptError):
                print("Interrupted by Ctrl-C. Stopping after writing partial solution.")
                break

    if not any_success and last_error is not None:
        elapsed = time.perf_counter() - started_at
        print(f"Total runtime: {elapsed:.2f}s")
        if isinstance(last_error, UserInterruptError):
            raise SystemExit(130)
        raise SystemExit(1)

    elapsed = time.perf_counter() - started_at
    print(f"Total runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
