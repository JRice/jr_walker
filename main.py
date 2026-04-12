import sys
from pathlib import Path
import argparse
import tomllib
from dataclasses import dataclass
import re
import sqlite3
import uuid

# Allow `python main.py` from repo root without installing the package.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jr_walker.validator import SubmissionValidator, ValidationError
from jr_walker.solver import SolverConfig, WarehouseSolver, PastRunAnalysis, load_best_past_analysis
from jr_walker.view import WarehouseState
from jr_walker.writer import write_actions
from jr_walker.analysis import build_and_store_solution_metadata


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
    lane_width: int = 3
    relocation_skus_to_relocate: list[int] | None = None
    max_time: int = 50000
    max_makespan: int | None = None
    max_plan_time_seconds: float = 600.0
    num_allowed_relocations: int = 10
    order_suggestion_gain_constant: float = 100.0
    lns_enabled: bool = True
    lns_iterations: int = 60
    lns_window_actions: int = 28
    lns_tail_fraction: float = 0.35
    lns_max_shift: int = 2
    output_dir: str = "output"
    log_path: str = "output/run.log"
    metadata_db_path: str = "output/solution_metadata.db"
    input_path: str = "docs/BIG_ORDER.txt"
    test_mode: bool = False
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

    raw_test_mode = solver_section.get("test_mode", run_config.test_mode)
    if not isinstance(raw_test_mode, bool):
        raise ValueError("solver.test_mode must be a boolean.")
    run_config.test_mode = raw_test_mode

    run_config.num_allowed_relocations = _require_int(
        solver_section.get("num_allowed_relocations", run_config.num_allowed_relocations),
        "solver.num_allowed_relocations",
        minimum=0,
    )
    run_config.order_suggestion_gain_constant = _require_number(
        solver_section.get("order_suggestion_gain_constant", run_config.order_suggestion_gain_constant),
        "solver.order_suggestion_gain_constant",
        minimum=0.0,
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
            max_makespan=run_config.max_makespan,
            max_plan_time_seconds=run_config.max_plan_time_seconds,
            output_path=Path(run_config.output_path) if run_config.output_path else temp_output_path,
            progress_every=50,
            log_path=Path(run_config.log_path),
            lane_width=run_config.lane_width,
            num_allowed_relocations=run_config.num_allowed_relocations,
            order_suggestion_gain_constant=run_config.order_suggestion_gain_constant,
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
    try:
        print("Building a fresh base solution guided by past analysis...")
        actions = solver.find_solution()

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

    except Exception as exc:
        solve_error = exc
        print(f"Solver/optimizer failed: {exc}")
        if actions:
            print("Writing best-known actions from current baseline...")
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
    if solve_error is not None:
        raise SystemExit(1)


def main():
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
    if run_config.test_mode:
        state.orders = state.orders[::10]

    if config_path.exists():
        print(
            "Loaded config from "
            f"{config_path} "
            f"(lane_width={run_config.lane_width}, "
            f"max_time={run_config.max_time}, "
            f"max_makespan={run_config.max_makespan}, "
            f"max_plan_time={run_config.max_plan_time_seconds:.1f}s, "
            f"forced_reloc_skus={run_config.relocation_skus_to_relocate})."
        )

    output_dir = Path(run_config.output_dir)
    output_prefix = "test_" if run_config.test_mode else ""
    temp_output_path = output_dir / f"{output_prefix}solution_latest.txt"

    past_analysis = PastRunAnalysis()
    db_path = Path(run_config.metadata_db_path)
    if db_path.exists():
        print("Loading analysis of the best existing solution from SQL database...")
        past_analysis = load_best_past_analysis(db_path, state.width, state.height, state.pallets)
        if past_analysis.run_id >= 0:
            print(f"Loaded analysis for Run #{past_analysis.run_id}: found {len(past_analysis.high_use_cells)} high-traffic cells to avoid.")

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


if __name__ == "__main__":
    main()
