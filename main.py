import sys
from pathlib import Path
import argparse
import tomllib
from dataclasses import dataclass
import re

# Allow `python main.py` from repo root without installing the package.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jr_walker.validator import ValidationError, validate_solution_file


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


def load_actions_from_solution(solution_path: Path) -> list[tuple[int, int, str, int, int]]:
    actions: list[tuple[int, int, str, int, int]] = []
    for line_no, raw in enumerate(solution_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid action row in {solution_path} at line {line_no}: {raw}")
        t_s, rid_s, action, x_s, y_s = parts[:5]
        try:
            action_row = (int(t_s), int(rid_s), action.lower(), int(x_s), int(y_s))
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric value in {solution_path} at line {line_no}: {raw}"
            ) from exc
        actions.append(action_row)
    actions.sort(key=lambda row: (row[0], row[1]))
    return actions


def find_best_existing_solution(output_dir: Path, *, test_mode: bool) -> Path | None:
    prefix = "test_solution_" if test_mode else "solution_"
    filename_pattern = re.compile(rf"^{re.escape(prefix)}(\d+)(?:_\d+)?\.txt$")
    candidates: list[tuple[int, float, Path]] = []
    for path in output_dir.glob(f"{prefix}*.txt"):
        stem = path.stem
        if "_F" in stem:
            continue
        match = filename_pattern.match(path.name)
        if not match:
            continue
        score = int(match.group(1))
        candidates.append((score, path.stat().st_mtime, path))

    if not candidates:
        return None
    candidates.sort(key=lambda row: (row[0], row[1]))
    return candidates[0][2]


def _parse_robot_key(raw_key: str) -> int:
    key = raw_key.strip().lower()
    if key.startswith("robot_"):
        key = key[len("robot_") :]
    if not key.isdigit():
        raise ValueError(f'Invalid robot key "{raw_key}" (expected e.g. "robot_0").')
    return int(key)


@dataclass
class RunConfig:
    role_plans_by_robot: dict[int, list[str]] | None = None
    lane_width: int = 3
    relocation_skus_to_relocate: list[int] | None = None
    max_time: int = 50000
    max_makespan: int | None = None
    max_plan_time_seconds: float = 600.0
    min_jobs_for_dock: int = 3
    lns_enabled: bool = True
    lns_iterations: int = 60
    lns_window_actions: int = 28
    lns_tail_fraction: float = 0.35
    lns_max_shift: int = 2


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


def load_run_config(config_path: Path, robot_count: int) -> RunConfig:
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    run_config = RunConfig()

    robots_section = data.get("robots")
    if robots_section is not None:
        if not isinstance(robots_section, dict):
            raise ValueError('If present, [robots] must be a table.')
        plans: dict[int, list[str]] = {}
        for raw_key, raw_roles in robots_section.items():
            robot_id = _parse_robot_key(str(raw_key))
            if robot_id < 0 or robot_id >= robot_count:
                raise ValueError(
                    f"Dispatch config references robot {robot_id}, but valid IDs are 0..{robot_count - 1}."
                )
            if not isinstance(raw_roles, list) or not all(isinstance(role, str) for role in raw_roles):
                raise ValueError(
                    f'Roles for "{raw_key}" must be an array of strings.'
                )
            cleaned = [role.strip().lower() for role in raw_roles if role.strip()]
            if not cleaned:
                raise ValueError(f'Roles for "{raw_key}" cannot be empty.')
            plans[robot_id] = cleaned

        missing = [rid for rid in range(robot_count) if rid not in plans]
        if missing:
            raise ValueError(f"Dispatch config is missing role plans for robots: {missing}")
        run_config.role_plans_by_robot = plans

    relocation_section = _get_table(data, "relocation")
    run_config.lane_width = _require_int(
        relocation_section.get("lane_width", run_config.lane_width),
        "relocation.lane_width",
        minimum=0,
    )
    raw_reloc_skus = relocation_section.get("skus_to_relocate", run_config.relocation_skus_to_relocate)
    if raw_reloc_skus is not None:
        if not isinstance(raw_reloc_skus, list) or not all(isinstance(v, int) for v in raw_reloc_skus):
            raise ValueError("relocation.skus_to_relocate must be an array of integers.")
        if any(v < 0 for v in raw_reloc_skus):
            raise ValueError("relocation.skus_to_relocate cannot contain negative SKU IDs.")
        run_config.relocation_skus_to_relocate = list(dict.fromkeys(raw_reloc_skus))

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

    return run_config


def main():
    parser = argparse.ArgumentParser(description="Build a warehouse action plan.")
    parser.add_argument("--input", default="docs/BIG_ORDER.txt", help="Path to BIG_ORDER-style input file.")
    parser.add_argument(
        "--validate-only",
        default=None,
        help="Validate an existing solution file and exit on first error.",
    )
    parser.add_argument("--output-dir", default="output", help="Directory to write solution files.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path. If omitted, uses solution_<makespan>.txt.",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=None,
        help="Optional override for solver.max_time (reservation horizon) from config.toml.",
    )
    parser.add_argument(
        "--log-path",
        default="output/run.log",
        help="Path for live runtime log output.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only keep every 10th order from the input worklist.",
    )
    parser.add_argument(
        "--config",
        default="docs/config.toml",
        help='TOML run config with per-robot role plans and relocation params. Default: docs/config.toml',
    )
    parser.add_argument(
        "--find-solution",
        action="store_true",
        help="Force building a fresh base solution before optimization.",
    )
    parser.add_argument(
        "--metadata-db-path",
        default="output/solution_metadata.db",
        help="SQLite DB path used to persist per-run solution metadata.",
    )
    parser.add_argument(
        "--metadata-run-id",
        type=int,
        default=None,
        help=(
            "Optional run ID for metadata persistence. "
            "If omitted, inferred from solution filename suffix (e.g. *_1) or auto-incremented."
        ),
    )
    args = parser.parse_args()

    if args.validate_only:
        try:
            final_state = validate_solution_file(
                Path(args.validate_only),
                worklist_path=Path(args.input),
            )
        except ValidationError as exc:
            print(str(exc))
            raise SystemExit(1)
        print(
            f"Validation passed. Fulfilled {final_state.fulfilled_orders}/{final_state.total_orders} "
            f"orders by timestep {final_state.next_timestep}."
        )
        return

    from jr_walker.solver import SolverConfig, WarehouseSolver
    from jr_walker.view import WarehouseState
    from jr_walker.writer import write_actions
    from jr_walker.analysis import build_and_store_solution_metadata

    state = WarehouseState(args.input)
    if args.test:
        state.orders = state.orders[::10]

    run_config = RunConfig()
    config_path = Path(args.config)
    if config_path.exists():
        try:
            run_config = load_run_config(
                config_path=config_path,
                robot_count=len(state.robots),
            )
        except Exception as exc:
            print(f"Failed to parse config {config_path}: {exc}")
            raise SystemExit(1)
        print(
            "Loaded config from "
            f"{config_path} "
            f"(lane_width={run_config.lane_width}, "
            f"max_time={run_config.max_time}, "
            f"max_makespan={run_config.max_makespan}, "
            f"max_plan_time={run_config.max_plan_time_seconds:.1f}s, "
            f"forced_reloc_skus={run_config.relocation_skus_to_relocate})."
        )
    else:
        print(f"Config file not found at {config_path}; using solver defaults.")

    if args.max_time is not None:
        run_config.max_time = args.max_time

    output_dir = Path(args.output_dir)
    output_prefix = "test_" if args.test else ""
    temp_output_path = output_dir / f"{output_prefix}solution_latest.txt"
    solver = WarehouseSolver(
        state,
        SolverConfig(
            max_time=run_config.max_time,
            max_makespan=run_config.max_makespan,
            max_plan_time_seconds=run_config.max_plan_time_seconds,
            output_path=Path(args.output) if args.output else temp_output_path,
            progress_every=50,
            log_path=Path(args.log_path),
            role_plans_by_robot=run_config.role_plans_by_robot,
            lane_width=run_config.lane_width,
            relocation_skus_to_relocate=run_config.relocation_skus_to_relocate,
            min_jobs_for_dock=run_config.min_jobs_for_dock,
            worklist_path=Path(args.input),
            lns_enabled=run_config.lns_enabled,
            lns_iterations=run_config.lns_iterations,
            lns_window_actions=run_config.lns_window_actions,
            lns_tail_fraction=run_config.lns_tail_fraction,
            lns_max_shift=run_config.lns_max_shift,
        ),
    )
    solve_error: Exception | None = None
    actions: list[tuple[int, int, str, int, int]] = []
    baseline_solution_path: Path | None = None
    try:
        if args.find_solution:
            print("Building a fresh base solution (--find-solution enabled)...")
            actions = solver.find_solution()
        else:
            baseline_solution_path = find_best_existing_solution(output_dir, test_mode=args.test)
            if baseline_solution_path is not None:
                print(f"Using existing base solution for optimization: {baseline_solution_path}")
                try:
                    actions = load_actions_from_solution(baseline_solution_path)
                except Exception as exc:
                    print(
                        f"Failed to load existing base solution {baseline_solution_path}: {exc}. "
                        "Falling back to fresh solve."
                    )
                    actions = []
                    baseline_solution_path = None

            if not actions:
                print("No reusable base solution found; building a fresh base solution...")
                actions = solver.find_solution()

        print(f"Running LNS optimization on {len(actions)} base actions...")
        actions = solver.optimize_actions(actions)
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

    makespan = max((t for t, _, _, _, _ in actions), default=-1)
    move_count = sum(1 for _, _, action, _, _ in actions if action == "move")

    if args.output:
        final_output_path = Path(args.output)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        name_prefix = f"{output_prefix}solution"
        if solve_error is not None:
            name_prefix = f"{output_prefix}partial_solution"
        final_output_path = make_unique_path(output_dir / f"{name_prefix}_{makespan}.txt")

    write_actions(actions, final_output_path)
    if not args.output and temp_output_path.exists():
        temp_output_path.unlink()

    metadata_run_id = None
    metadata_db_path = Path(args.metadata_db_path)
    try:
        metadata_run_id = build_and_store_solution_metadata(
            solution_path=final_output_path,
            worklist_path=Path(args.input),
            metadata_db_path=metadata_db_path,
            metadata_run_id=args.metadata_run_id,
        )
    except Exception as analysis_exc:
        print(f"Metadata persistence failed: {analysis_exc}")

    print(f"Wrote {len(actions)} actions to {final_output_path}")
    print(f"Move count: {move_count}")
    print(f"Plan makespan: {makespan} timesteps")
    print(f"Orders used: {len(state.orders)}")
    if metadata_run_id is not None:
        print(f"Wrote metadata to {metadata_db_path} (run_id={metadata_run_id})")
    if solve_error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
