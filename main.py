import sys
from pathlib import Path
import argparse

# Allow `python main.py` from repo root without installing the package.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jr_walker.solver import SolverConfig, WarehouseSolver
from jr_walker.view import WarehouseState
from jr_walker.writer import write_actions
from jr_walker.analysis import analysis_output_path_for_solution, solution_analysis


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


def main():
    parser = argparse.ArgumentParser(description="Build a warehouse action plan.")
    parser.add_argument("--input", default="data/BIG_ORDER.txt", help="Path to BIG_ORDER-style input file.")
    parser.add_argument(
        "--analyze-only",
        default=None,
        help="Analyze an existing solution file and exit.",
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
    args = parser.parse_args()

    if args.analyze_only:
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

    state = WarehouseState(args.input)
    output_dir = Path(args.output_dir)
    temp_output_path = output_dir / "solution_latest.txt"
    solver = WarehouseSolver(
        state,
        SolverConfig(
            max_time=args.max_time,
            output_path=Path(args.output) if args.output else temp_output_path,
            progress_every=50,
            log_path=Path(args.log_path),
        ),
    )
    _, actions = solver.solve()

    makespan = max((t for t, _, _, _, _ in actions), default=-1)
    move_count = sum(1 for _, _, action, _, _ in actions if action == "move")

    if args.output:
        final_output_path = Path(args.output)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = make_unique_path(output_dir / f"solution_{makespan}.txt")

    write_actions(actions, final_output_path)
    if not args.output and temp_output_path.exists():
        temp_output_path.unlink()

    analysis_path = analysis_output_path_for_solution(final_output_path, output_dir=output_dir)
    written_analysis_path = solution_analysis(
        solution_path=final_output_path,
        worklist_path=Path(args.input),
        output_path=analysis_path,
    )

    print(f"Wrote {len(actions)} actions to {final_output_path}")
    print(f"Move count: {move_count}")
    print(f"Plan makespan: {makespan} timesteps")
    print(f"Wrote analysis to {written_analysis_path}")


if __name__ == "__main__":
    main()
