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


def main():
    parser = argparse.ArgumentParser(description="Build a warehouse action plan.")
    parser.add_argument("--input", default="data/BIG_ORDER.txt", help="Path to BIG_ORDER-style input file.")
    parser.add_argument(
        "--output", default="output/solution.txt", help="Path to write action plan (.txt)."
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=50000,
        help="Reservation horizon in timesteps.",
    )
    args = parser.parse_args()

    state = WarehouseState(args.input)
    solver = WarehouseSolver(
        state,
        SolverConfig(
            max_time=args.max_time,
            output_path=Path(args.output),
            progress_every=50,
        ),
    )
    output_path, actions = solver.solve()

    makespan = max((t for t, _, _, _, _ in actions), default=-1)
    print(f"Wrote {len(actions)} actions to {output_path}")
    print(f"Plan makespan: {makespan} timesteps")


if __name__ == "__main__":
    main()
