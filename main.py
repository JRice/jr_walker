import sys
from pathlib import Path
import argparse
from datetime import datetime
import tomllib

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


def append_leaderboard_entry(
    leaderboard_path: Path,
    *,
    run_mode: str,
    input_path: Path,
    total_orders: int,
    makespan: int,
    move_count: int,
    solution_path: Path,
    analysis_path: Path,
    algorithm_update: str,
) -> None:
    leaderboard_path = Path(leaderboard_path)
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)

    if not leaderboard_path.exists():
        leaderboard_path.write_text("# Leaderboard\n\n", encoding="utf-8")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    note = algorithm_update.strip() or "TODO: describe latest algorithm update"
    lines = [
        f"## Run {timestamp}",
        f"- Mode: {run_mode}",
        f"- Input: {input_path}",
        f"- Orders used: {total_orders}",
        f"- Score (makespan): {makespan}",
        f"- Move count: {move_count}",
        f"- Solution: {solution_path}",
        f"- Analysis: {analysis_path}",
        f"- Latest algorithm update: {note}",
        "- Validated: TODO",
        "",
    ]
    with leaderboard_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def _parse_robot_key(raw_key: str) -> int:
    key = raw_key.strip().lower()
    if key.startswith("robot_"):
        key = key[len("robot_") :]
    if not key.isdigit():
        raise ValueError(f'Invalid robot key "{raw_key}" (expected e.g. "robot_0").')
    return int(key)


def load_dispatch_role_plans(config_path: Path, robot_count: int) -> dict[int, list[str]]:
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    robots_section = data.get("robots")
    if not isinstance(robots_section, dict):
        raise ValueError('Dispatch config must contain a [robots] table.')

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

    return plans


def main():
    parser = argparse.ArgumentParser(description="Build a warehouse action plan.")
    parser.add_argument("--input", default="docs/BIG_ORDER.txt", help="Path to BIG_ORDER-style input file.")
    parser.add_argument(
        "--analyze-only",
        default=None,
        help="Analyze an existing solution file and exit.",
    )
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
        default=13000,
        help="Reservation horizon in timesteps.",
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
        "--run-note",
        default="",
        help="Optional note describing the latest algorithm change for leaderboard.md.",
    )
    parser.add_argument(
        "--dispatch-config",
        default=None,
        help='Optional TOML file with per-robot role plans, e.g. [robots] robot_0=["relocate_pallet","loop","deliver_easy","deliver_hard"].',
    )
    args = parser.parse_args()

    if args.analyze_only:
        from jr_walker.analysis import analysis_output_path_for_solution, solution_analysis

        analysis_path = analysis_output_path_for_solution(
            Path(args.analyze_only), output_dir=Path(args.output_dir)
        )
        written = solution_analysis(
            solution_path=Path(args.analyze_only),
            worklist_path=Path(args.input),
            output_path=analysis_path,
        )
        print(f"Wrote analysis to {written}")
        return

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
    from jr_walker.analysis import analysis_output_path_for_solution, solution_analysis

    state = WarehouseState(args.input)
    if args.test:
        state.orders = state.orders[::10]

    role_plans_by_robot = None
    if args.dispatch_config:
        config_path = Path(args.dispatch_config)
        if not config_path.exists():
            print(f"Dispatch config not found: {config_path}")
            raise SystemExit(1)
        try:
            role_plans_by_robot = load_dispatch_role_plans(
                config_path=config_path,
                robot_count=len(state.robots),
            )
        except Exception as exc:
            print(f"Failed to parse dispatch config {config_path}: {exc}")
            raise SystemExit(1)
        print(f"Loaded dispatch role plans from {config_path} for {len(role_plans_by_robot)} robots.")

    output_dir = Path(args.output_dir)
    output_prefix = "test_" if args.test else ""
    temp_output_path = output_dir / f"{output_prefix}solution_latest.txt"
    solver = WarehouseSolver(
        state,
        SolverConfig(
            max_time=args.max_time,
            output_path=Path(args.output) if args.output else temp_output_path,
            progress_every=50,
            log_path=Path(args.log_path),
            role_plans_by_robot=role_plans_by_robot,
        ),
    )
    solve_error: Exception | None = None
    try:
        _, actions = solver.solve()
    except Exception as exc:
        solve_error = exc
        print(f"Solver failed: {exc}")
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

    written_analysis_path = None
    try:
        analysis_path = analysis_output_path_for_solution(final_output_path, output_dir=output_dir)
        written_analysis_path = solution_analysis(
            solution_path=final_output_path,
            worklist_path=Path(args.input),
            output_path=analysis_path,
        )
    except Exception as analysis_exc:
        print(f"Analysis failed: {analysis_exc}")

    if solve_error is None and written_analysis_path is not None:
        append_leaderboard_entry(
            output_dir / "leaderboard.md",
            run_mode="test-10x" if args.test else "full",
            input_path=Path(args.input),
            total_orders=len(state.orders),
            makespan=makespan,
            move_count=move_count,
            solution_path=final_output_path,
            analysis_path=written_analysis_path,
            algorithm_update=args.run_note,
        )

    print(f"Wrote {len(actions)} actions to {final_output_path}")
    print(f"Move count: {move_count}")
    print(f"Plan makespan: {makespan} timesteps")
    print(f"Orders used: {len(state.orders)}")
    if written_analysis_path is not None:
        print(f"Wrote analysis to {written_analysis_path}")
    if solve_error is None and written_analysis_path is not None:
        print(f"Updated leaderboard: {output_dir / 'leaderboard.md'}")
    if solve_error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
