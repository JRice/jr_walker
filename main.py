from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
import tomllib

# Allow `python main.py` from repo root without installing the package.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jr_walker.cooperative_solver import CooperativeWarehouseSolver
from jr_walker.solver import PastRunAnalysis, SolverConfig
from jr_walker.validator import SubmissionValidator, ValidationError
from jr_walker.view import WarehouseState
from jr_walker.writer import write_actions


class UserInterruptError(ValidationError):
    """Raised when planning is interrupted by Ctrl-C after saving partial state."""


@dataclass
class RunConfig:
    input_path: str = "config/BIG_ORDER.txt"
    output_dir: str = "output"
    media_dir: str = "media"
    log_path: str = "output/run.log"
    run_number_path: str = "config/run_number.txt"
    output_path: str | None = None
    stride: int = 1
    max_ticks: int = 16_000
    max_runtime_minutes: float = 60.0
    max_time: int = 50_000
    strict_no_swap: bool = False
    conveyor_wait_stall_limit: int = 24
    max_idle_ticks: int = 24
    setup_hotspots: list[tuple[int, int]] = field(default_factory=lambda: [(15, 0), (35, 0)])
    progress_order_interval: int = 100
    progress_tick_interval: int = 1000
    lns_enabled: bool = False
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


def load_run_config(config_path: Path) -> RunConfig:
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    config = RunConfig()
    paths = _get_table(data, "paths")
    limits = _get_table(data, "limits")
    solver = _get_table(data, "solver")
    relocation = _get_table(data, "relocation")
    progress = _get_table(data, "progress")
    lns = _get_table(data, "lns")

    config.input_path = str(paths.get("input_path", config.input_path))
    config.output_dir = str(paths.get("output_dir", config.output_dir))
    config.media_dir = str(paths.get("media_dir", config.media_dir))
    config.log_path = str(paths.get("log_path", config.log_path))
    config.run_number_path = str(paths.get("run_number_path", config.run_number_path))
    raw_output_path = paths.get("output_path")
    if raw_output_path is not None:
        config.output_path = str(raw_output_path)

    raw_stride = limits.get("stride", limits.get("every_n_orders", config.stride))
    config.stride = _require_int(raw_stride, "limits.stride", minimum=1)
    raw_max_ticks = limits.get("max_ticks", limits.get("max_makespan", config.max_ticks))
    config.max_ticks = _require_int(raw_max_ticks, "limits.max_ticks", minimum=1)
    raw_runtime = limits.get("max_runtime_minutes")
    if raw_runtime is None:
        # Backward-compatible read from old seconds field.
        old_seconds = limits.get("max_plan_time")
        if old_seconds is not None:
            raw_runtime = float(old_seconds) / 60.0
        else:
            raw_runtime = config.max_runtime_minutes
    config.max_runtime_minutes = _require_number(
        raw_runtime,
        "limits.max_runtime_minutes",
        minimum=0.1,
    )

    config.max_time = _require_int(solver.get("max_time", config.max_time), "solver.max_time", minimum=2)
    raw_strict = solver.get("strict_no_swap", config.strict_no_swap)
    if not isinstance(raw_strict, bool):
        raise ValueError("solver.strict_no_swap must be a boolean.")
    config.strict_no_swap = raw_strict
    config.conveyor_wait_stall_limit = _require_int(
        solver.get("conveyor_wait_stall_limit", config.conveyor_wait_stall_limit),
        "solver.conveyor_wait_stall_limit",
        minimum=1,
    )
    config.max_idle_ticks = _require_int(
        solver.get("max_idle_ticks", config.max_idle_ticks),
        "solver.max_idle_ticks",
        minimum=1,
    )

    raw_hotspots = relocation.get("setup_hotspots", config.setup_hotspots)
    if not isinstance(raw_hotspots, list):
        raise ValueError("relocation.setup_hotspots must be a list of [x, y] pairs.")
    parsed_hotspots: list[tuple[int, int]] = []
    for idx, cell in enumerate(raw_hotspots):
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            raise ValueError(f"relocation.setup_hotspots[{idx}] must be [x, y].")
        x = _require_int(cell[0], f"relocation.setup_hotspots[{idx}][0]", minimum=0)
        y = _require_int(cell[1], f"relocation.setup_hotspots[{idx}][1]", minimum=0)
        parsed_hotspots.append((x, y))
    config.setup_hotspots = parsed_hotspots

    config.progress_order_interval = _require_int(
        progress.get("order_interval", config.progress_order_interval),
        "progress.order_interval",
        minimum=1,
    )
    config.progress_tick_interval = _require_int(
        progress.get("tick_interval", config.progress_tick_interval),
        "progress.tick_interval",
        minimum=1,
    )

    raw_lns_enabled = lns.get("enabled", config.lns_enabled)
    if not isinstance(raw_lns_enabled, bool):
        raise ValueError("lns.enabled must be a boolean.")
    config.lns_enabled = raw_lns_enabled
    config.lns_iterations = _require_int(
        lns.get("iterations", config.lns_iterations),
        "lns.iterations",
        minimum=0,
    )
    config.lns_window_actions = _require_int(
        lns.get("window_actions", config.lns_window_actions),
        "lns.window_actions",
        minimum=1,
    )
    config.lns_tail_fraction = _require_number(
        lns.get("tail_fraction", config.lns_tail_fraction),
        "lns.tail_fraction",
        minimum=0.0,
    )
    if config.lns_tail_fraction > 1.0:
        raise ValueError("lns.tail_fraction must be <= 1.0.")
    config.lns_max_shift = _require_int(
        lns.get("max_shift", config.lns_max_shift),
        "lns.max_shift",
        minimum=1,
    )
    return config


def next_run_id(run_number_path: Path) -> int:
    run_number_path.parent.mkdir(parents=True, exist_ok=True)
    if not run_number_path.exists():
        run_number_path.write_text("1\n", encoding="utf-8")
        return 1

    raw = run_number_path.read_text(encoding="utf-8").strip()
    if not raw:
        run_number_path.write_text("1\n", encoding="utf-8")
        return 1
    try:
        current = int(raw)
    except ValueError:
        current = 0
    nxt = max(1, current + 1)
    run_number_path.write_text(f"{nxt}\n", encoding="utf-8")
    return nxt


def _save_fulfill_rate_plot(
    actions: list[tuple[int, int, str, int, int]],
    run_id: int,
    *,
    max_tick: int | None = None,
    fulfilled_order_count: int | None = None,
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

    media_dir.mkdir(parents=True, exist_ok=True)
    if max_tick is None or fulfilled_order_count is None:
        output_path = media_dir / f"fulfill_rate_run_{int(run_id)}.png"
        title = f"Fulfill Rate Over Time (run_id={int(run_id)})"
    else:
        output_path = media_dir / (
            f"run{int(run_id)}_{int(max_tick)}tick_{int(fulfilled_order_count)}order_rate.png"
        )
        title = (
            f"Fulfill Rate Over Time (run={int(run_id)}, "
            f"tick={int(max_tick)}, fulfilled={int(fulfilled_order_count)})"
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(x_points, y_points, where="post", linewidth=2.0)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative Fulfilled Orders")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _build_solver_config(run_config: RunConfig, *, temp_output_path: Path) -> SolverConfig:
    solver_config = SolverConfig(
        max_time=run_config.max_time,
        max_makespan=run_config.max_ticks,
        max_plan_time_seconds=run_config.max_runtime_minutes * 60.0,
        progress_every=max(1, int(run_config.progress_order_interval)),
        output_path=Path(run_config.output_path) if run_config.output_path else temp_output_path,
        log_path=Path(run_config.log_path),
        strict_no_swap=run_config.strict_no_swap,
        setup_hotspots=list(run_config.setup_hotspots),
        single_nest_conveyor_mode=True,
        conveyor_wait_stall_limit=run_config.conveyor_wait_stall_limit,
        worklist_path=Path(run_config.input_path),
        lns_enabled=run_config.lns_enabled,
        lns_iterations=run_config.lns_iterations,
        lns_window_actions=run_config.lns_window_actions,
        lns_tail_fraction=run_config.lns_tail_fraction,
        lns_max_shift=run_config.lns_max_shift,
    )
    setattr(solver_config, "max_idle_ticks", int(run_config.max_idle_ticks))
    setattr(solver_config, "progress_order_interval", int(run_config.progress_order_interval))
    setattr(solver_config, "progress_tick_interval", int(run_config.progress_tick_interval))
    return solver_config


def _format_output_path(
    *,
    output_dir: Path,
    run_id: int,
    makespan: int,
    fulfilled_orders: int,
    explicit_output_path: str | None,
) -> Path:
    if explicit_output_path:
        return Path(explicit_output_path)
    return output_dir / f"run{int(run_id)}_{int(makespan)}tick_{int(fulfilled_orders)}order_solution.txt"


def main() -> None:
    started_at = time.perf_counter()
    parser = argparse.ArgumentParser(description="Build a cooperative multi-nest warehouse action plan.")
    parser.add_argument(
        "--config",
        default="config/config.toml",
        help="TOML run config path. Default: config/config.toml",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        raise SystemExit(1)

    try:
        run_config = load_run_config(config_path)
    except Exception as exc:
        print(f"Failed to parse config {config_path}: {exc}")
        raise SystemExit(1)

    run_id = next_run_id(Path(run_config.run_number_path))
    print(
        f"Loaded {config_path} | run_id={run_id} stride={run_config.stride} max_ticks={run_config.max_ticks} "
        f"max_runtime_minutes={run_config.max_runtime_minutes:.2f} nests={run_config.setup_hotspots}"
    )

    state = WarehouseState(run_config.input_path)
    if run_config.stride > 1:
        state.orders = state.orders[:: run_config.stride]
    print(f"Using {len(state.orders)} orders from {run_config.input_path}")

    output_dir = Path(run_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_dir / f"run{run_id}_latest.txt"

    solver = CooperativeWarehouseSolver(
        state,
        _build_solver_config(run_config, temp_output_path=temp_output_path),
        past_analysis=PastRunAnalysis(),
    )

    actions: list[tuple[int, int, str, int, int]] = []
    solve_error: Exception | None = None
    try:
        actions = solver.find_solution()
    except KeyboardInterrupt:
        solve_error = UserInterruptError("Interrupted by Ctrl-C; writing partial output.")
        actions = solver.actions.sorted_actions()
    except Exception as exc:
        solve_error = exc
        actions = solver.actions.sorted_actions()
        print(f"Solver failed: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())

    fulfilled_orders = len(getattr(solver, "_completed_order_indices", set()))
    makespan = max((t for t, _, _, _, _ in actions), default=-1)
    output_path = _format_output_path(
        output_dir=output_dir,
        run_id=run_id,
        makespan=makespan,
        fulfilled_orders=fulfilled_orders,
        explicit_output_path=run_config.output_path,
    )
    write_actions(actions, output_path)
    print(f"Wrote {len(actions)} actions to {output_path}")

    if solve_error is None:
        validator = SubmissionValidator(worklist_path=run_config.input_path)
        try:
            final_state = validator.validate_file(output_path)
            fulfilled_orders = int(final_state.fulfilled_orders)
            print(
                f"Validator passed: fulfilled={final_state.fulfilled_orders}/{final_state.total_orders} "
                f"next_timestep={final_state.next_timestep}"
            )
            if final_state.fulfilled_orders == final_state.total_orders:
                makespan = max((t for t, _, _, _, _ in actions), default=-1)
                plot_path = _save_fulfill_rate_plot(
                    actions,
                    run_id=run_id,
                    max_tick=makespan,
                    fulfilled_order_count=fulfilled_orders,
                    media_dir=Path(run_config.media_dir),
                )
                print(f"Wrote fulfill-rate plot to {plot_path}")
        except Exception as exc:
            print(f"Validation failed for written output: {exc}")
            solve_error = exc
    else:
        print("Skipped validator replay because planning failed; partial output was still written.")

    elapsed = time.perf_counter() - started_at
    print(f"Total runtime: {elapsed:.2f}s")

    if solve_error is not None:
        if isinstance(solve_error, UserInterruptError):
            raise SystemExit(130)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
