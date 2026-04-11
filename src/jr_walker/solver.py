import collections
import json
import random
import sqlite3
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Deque, Dict, List, Tuple

import numpy as np

from jr_walker.logic import EdgeAwareOrderScorer, OrderOptimizer
from jr_walker.planner import adjacent_cells
from jr_walker.planner import ReservationPlanner
from jr_walker.scheduler import GreedyScheduler
from jr_walker.sim import ActionLog, RobotState
from jr_walker.validator import SubmissionValidator, ValidationError
from jr_walker.writer import write_actions

ROLE_DELIVER = "deliver"
ROLE_RELOCATE_PALLET = "relocate_pallet"
DELIVER_EASY = "easy"
DELIVER_HARD = "hard"
ROLE_LOOP = "loop"
ROLE_DELIVER_EASY = f"{ROLE_DELIVER}_{DELIVER_EASY}"
ROLE_DELIVER_HARD = f"{ROLE_DELIVER}_{DELIVER_HARD}"

# Planned delivery hotspot anchors.
FULFILL_HOT_SPOTS: List[Tuple[int, int]] = [
    (20, 0),
    (40, 0),
    (20, 39),
    (40, 39),
    (0, 20),
    (59, 20),
]

KNIGHT_OFFSETS: List[Tuple[int, int]] = [
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
]

BUCKET_TO_HOTSPOT: Dict[str, Tuple[int, int]] = {
    "left_edge": (0, 20),
    "right_edge": (59, 20),
    "top_x0_29": (20, 0),
    "top_x30_59": (40, 0),
    "bottom_x0_29": (20, 39),
    "bottom_x30_59": (40, 39),
}


@dataclass
class RelocationJob:
    sku: int
    bucket: str
    hotspot: Tuple[int, int]
    score: float
    placement_offset: Tuple[int, int] = (0, 0)
    preferred_target_xy: Tuple[int, int] | None = None
    attempts: int = 0
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class PalletMove:
    old_xy: Tuple[int, int]
    new_xy: Tuple[int, int]
    dock_t: int
    undock_t: int


@dataclass
class PlannedOrder:
    order_idx: int
    items: collections.Counter
    estimated_cost: float = float("inf")

    def estimate_cost(self, scorer: EdgeAwareOrderScorer) -> float:
        self.estimated_cost = scorer.estimate_order_cost(self.items)
        return self.estimated_cost


@dataclass
class RobotFulfillEvent:
    timestep: int
    robot_id: int
    order_id: int | None
    skus: set[int]


@dataclass
class RobotRoleCursor:
    roles: List[str]
    next_index: int = 0
    loop_index: int | None = None


@dataclass
class SolverConfig:
    max_time: int = 50000
    max_makespan: int | None = None
    max_plan_time_seconds: float = 600.0
    progress_every: int = 50
    output_path: Path = Path("output/solution.txt")
    initial_relocate_dispatches: int = 8
    relocate_pallet_probability: float = 0.0
    random_seed: int = 7
    max_delivery_order_attempts: int = 40
    delivery_candidate_window: int = 160
    path_step_limit: int = 350
    relocate_stand_candidate_limit: int = 6
    relocate_target_candidate_limit: int = 8
    relocation_analysis_path: Path | None = None
    relocation_top_skus: int = 8
    relocation_min_lift: float = 0.08
    relocation_max_attempts_per_sku: int = 5
    relocation_skus_to_relocate: List[int] | None = None
    lane_width: int = 3
    min_jobs_for_dock: int = 3
    log_path: Path | None = None
    dispatch_log_every: int = 1
    worklist_path: Path = Path("docs/BIG_ORDER.txt")
    lns_enabled: bool = True
    lns_iterations: int = 60
    lns_window_actions: int = 28
    lns_tail_fraction: float = 0.35
    lns_max_shift: int = 2
    forced_dock_max_attempts_per_robot_sku: int = 3
    forced_dock_cooldown_dispatches: int = 25
    role_plans_by_robot: Dict[int, List[str]] | None = None


class WarehouseSolver:
    def __init__(self, warehouse_state, config: SolverConfig | None = None):
        self.state = warehouse_state
        self.config = config or SolverConfig()
        self.rng = random.Random(self.config.random_seed)

        self.robots: List[RobotState] = [
            RobotState(id=rid, x=x, y=y) for rid, (x, y) in enumerate(self.state.robots)
        ]
        self.orders: List[PlannedOrder] = [
            PlannedOrder(order_idx=i, items=collections.Counter(order))
            for i, order in enumerate(self.state.orders)
        ]
        self.actions = ActionLog()
        self.planner = ReservationPlanner(
            static_blocked=(self.state.grid == 2),
            width=self.state.width,
            height=self.state.height,
            max_time=self.config.max_time,
        )
        self.scheduler = GreedyScheduler(
            width=self.state.width,
            height=self.state.height,
            pallets=self.state.pallets,
        )
        self.edge_scorer = EdgeAwareOrderScorer(
            scheduler=self.scheduler,
            width=self.state.width,
            height=self.state.height,
            hot_spots=FULFILL_HOT_SPOTS,
        )
        self._metadata_db_path = self.config.relocation_analysis_path
        if self._metadata_db_path is None:
            self._metadata_db_path = self._find_default_metadata_db_path()
        self._metadata_use_by_cell: Dict[Tuple[int, int], int] = {}
        self._metadata_sku_cells: Dict[int, List[Tuple[int, int, int]]] = {}
        self._metadata_high_use_cells: set[Tuple[int, int]] = set()
        self._load_relocation_metadata()
        self.travel_lane_cells: set[Tuple[int, int]] = self._build_travel_lane_cells(
            lane_width=self.config.lane_width
        )

        # Stable pallet IDs so we can track moved pallets by SKU.
        self.pallet_by_id: Dict[int, Dict[str, int]] = {}
        self.pallet_id_by_coord: Dict[Tuple[int, int], int] = {}
        self.pallet_initial_xy: Dict[int, Tuple[int, int]] = {}
        self.pallet_moves: Dict[int, List[PalletMove]] = collections.defaultdict(list)
        for pallet_id, ((px, py), sku) in enumerate(self.state.pallets.items()):
            self.pallet_by_id[pallet_id] = {"sku": sku, "x": px, "y": py}
            self.pallet_id_by_coord[(px, py)] = pallet_id
            self.pallet_initial_xy[pallet_id] = (px, py)

        self.relocated_skus: set[int] = set()
        self.relocated_pallet_targets: set[Tuple[int, int]] = set()
        sku_counter: collections.Counter = collections.Counter()
        for order in self.orders:
            sku_counter.update(order.items)
        self.skus_by_demand: List[int] = [sku for sku, _ in sku_counter.most_common()]
        self.relocation_plan: Deque[RelocationJob] = self._build_relocation_plan()

        self.role_plans_by_robot = self._normalize_role_plans(
            self.config.role_plans_by_robot or {}
        )
        self._role_cursors_by_robot: Dict[int, RobotRoleCursor] = {}
        for robot_id, roles in self.role_plans_by_robot.items():
            loop_index = None
            for idx, role in enumerate(roles):
                if role == ROLE_LOOP:
                    loop_index = idx
                    break
            self._role_cursors_by_robot[robot_id] = RobotRoleCursor(
                roles=list(roles),
                next_index=0,
                loop_index=loop_index,
            )

        self._next_delivery_strategy_by_robot: Dict[int, str] = {
            robot.id: DELIVER_EASY for robot in self.robots
        }
        self._initial_relocate_assigned = 0
        self._dispatch_floor_t = -1
        self._warmup_barrier_applied = False
        self._lookahead_relocation_seeded_skus: set[int] = set()
        self._plan_started_monotonic = 0.0
        self._forced_dock_failures: Dict[Tuple[int, int], int] = collections.defaultdict(int)
        self._forced_dock_cooldown_until_dispatch: Dict[Tuple[int, int], int] = {}
        self._log_handle = None

    def _select_best_non_test_run_id(self, conn: sqlite3.Connection) -> int | None:
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

    def _load_relocation_metadata(self) -> None:
        path = self._metadata_db_path
        if path is None:
            return
        path = Path(path)
        if not path.exists():
            return

        conn = sqlite3.connect(path)
        try:
            try:
                run_id = self._select_best_non_test_run_id(conn)
                if run_id is None:
                    return
                use_rows = conn.execute(
                    "SELECT x, y, use_score FROM cell_metadata WHERE run_id = ?",
                    (run_id,),
                ).fetchall()
                for x, y, use_score in use_rows:
                    self._metadata_use_by_cell[(int(x), int(y))] = int(use_score)

                sku_rows = conn.execute(
                    "SELECT x, y, sku, count FROM cell_sku_flow WHERE run_id = ?",
                    (run_id,),
                ).fetchall()
                for x, y, sku, count in sku_rows:
                    self._metadata_sku_cells.setdefault(int(sku), []).append(
                        (int(x), int(y), int(count))
                    )
                for sku, rows in self._metadata_sku_cells.items():
                    rows.sort(
                        key=lambda row: (
                            -row[2],
                            self._metadata_use_by_cell.get((row[0], row[1]), 0),
                            row[1],
                            row[0],
                        )
                    )

                # Reinforce corridor detection from metadata "use" values while ignoring currently
                # occupied pallet cells (they can trivially dominate use due to static occupancy).
                dynamic_use_values: List[int] = []
                for (x, y), use_score in self._metadata_use_by_cell.items():
                    if use_score <= 0:
                        continue
                    if (x, y) in self.scheduler.pallets:
                        continue
                    dynamic_use_values.append(use_score)
                if dynamic_use_values:
                    dynamic_use_values.sort()
                    idx = int((len(dynamic_use_values) - 1) * 0.85)
                    cutoff = dynamic_use_values[max(0, idx)]
                    for (x, y), use_score in self._metadata_use_by_cell.items():
                        if use_score < cutoff:
                            continue
                        if (x, y) in self.scheduler.pallets:
                            continue
                        self._metadata_high_use_cells.add((x, y))
            except sqlite3.Error:
                return
        finally:
            conn.close()

    def solve(self) -> Tuple[Path, List[Tuple[int, int, str, int, int]]]:
        self._open_log()
        try:
            base_actions = self._find_solution_actions_core()
            sorted_actions = self._optimize_actions_core(base_actions)
            output_path = write_actions(sorted_actions, self.config.output_path)
            makespan = max((t for t, _, _, _, _ in sorted_actions), default=-1)
            self._log(
                f"solve_end actions={len(sorted_actions)} makespan={makespan} output={output_path}"
            )
            return output_path, sorted_actions
        finally:
            self._close_log()

    def find_solution(self) -> List[Tuple[int, int, str, int, int]]:
        self._open_log()
        try:
            return self._find_solution_actions_core()
        finally:
            self._close_log()

    def optimize_actions(
        self, actions: List[Tuple[int, int, str, int, int]]
    ) -> List[Tuple[int, int, str, int, int]]:
        self._open_log()
        try:
            seeded = self._try_suffix_replan_for_hot_sku(list(actions))
            return self._optimize_actions_core(seeded)
        finally:
            self._close_log()

    def _find_solution_actions_core(self) -> List[Tuple[int, int, str, int, int]]:
        """
        Algorithm: greedy event-loop dispatcher.
        Pattern: orchestrator loop that repeatedly assigns the next robot/task pair.
        """
        self._plan_started_monotonic = time.monotonic()
        remaining_orders = self._build_ranked_order_queue()
        self._recalculate_order_costs(remaining_orders)
        total_orders = len(remaining_orders)
        completed = 0
        dispatch_count = 0
        self._log_solve_start(total_orders)

        while remaining_orders:
            self._check_global_limits_or_raise(remaining_orders)
            self._inject_lookahead_relocation_jobs(remaining_orders)
            robot = self._next_available_robot()
            handled = self._execute_dispatch(
                robot=robot,
                remaining_orders=remaining_orders,
                completed=completed,
                total_orders=total_orders,
                dispatch_number=dispatch_count + 1,
            )
            if not handled:
                raise RuntimeError("Dispatcher could not assign a feasible next task.")

            new_completed = total_orders - len(remaining_orders)
            if new_completed != completed:
                completed = new_completed
                self._maybe_log_progress(
                    completed=completed,
                    total_orders=total_orders,
                    dispatch_count=dispatch_count + 1,
                )
            self._check_global_limits_or_raise(remaining_orders)

            dispatch_count += 1

        sorted_actions = self.actions.sorted_actions()
        sorted_actions = self._repair_idle_wait_conflicts(sorted_actions)
        makespan = max((t for t, _, _, _, _ in sorted_actions), default=-1)
        self._log(
            f"find_solution_end actions={len(sorted_actions)} makespan={makespan}"
        )
        return sorted_actions

    def _optimize_actions_core(
        self, actions: List[Tuple[int, int, str, int, int]]
    ) -> List[Tuple[int, int, str, int, int]]:
        """
        Pattern: optimization pipeline.
        Step 1 repairs known validity issues, Step 2 applies local-search improvement (LNS).
        """
        self._plan_started_monotonic = time.monotonic()
        repaired = self._repair_idle_wait_conflicts(list(actions))
        improved = self._lns_improve_actions(repaired)
        makespan = max((t for t, _, _, _, _ in improved), default=-1)
        self._log(
            f"optimize_end actions={len(improved)} makespan={makespan}"
        )
        return improved

    def _try_suffix_replan_for_hot_sku(
        self, actions: List[Tuple[int, int, str, int, int]]
    ) -> List[Tuple[int, int, str, int, int]]:
        threshold = max(2, int(self.config.min_jobs_for_dock))
        candidate = self._find_hot_sku_streak_candidate(actions, threshold)
        if candidate is None:
            self._log("suffix_replan_skip reason=no_hot_sku_streak")
            return actions

        robot_id, sku, streak_len, streak_start_t = candidate
        target_t = streak_start_t - 1
        snapshot = self._latest_clean_snapshot_before(actions, target_t)
        if snapshot is None:
            self._log(
                f"suffix_replan_skip reason=no_clean_cut robot={robot_id} sku={sku} target_t={target_t}"
            )
            return actions

        cut_t, robot_positions, pallet_map, fulfilled_order_ids = snapshot
        if cut_t < 0:
            self._log(
                f"suffix_replan_skip reason=clean_cut_before_start robot={robot_id} sku={sku}"
            )
            return actions
        remaining_orders = [
            collections.Counter(self.orders[oid].items)
            for oid in range(len(self.orders))
            if oid not in fulfilled_order_ids
        ]
        if not remaining_orders:
            self._log("suffix_replan_skip reason=no_remaining_orders")
            return actions

        suffix_state = self._build_suffix_state(
            robot_positions=robot_positions,
            pallet_map=pallet_map,
            remaining_orders=remaining_orders,
        )
        suffix_plans = self._build_suffix_role_plans(robot_id=robot_id, sku=sku)
        suffix_config = replace(
            self.config,
            role_plans_by_robot=suffix_plans,
            relocation_skus_to_relocate=[sku],
            relocation_analysis_path=None,
            lns_enabled=False,
            log_path=None,
        )
        suffix_solver = WarehouseSolver(suffix_state, suffix_config)
        try:
            suffix_actions = suffix_solver.find_solution()
        except Exception as exc:
            self._log(
                f"suffix_replan_skip reason=suffix_solve_failed robot={robot_id} sku={sku} error={exc}"
            )
            return actions

        shifted_suffix = [(t + cut_t + 1, rid, act, x, y) for (t, rid, act, x, y) in suffix_actions]
        prefix = [row for row in actions if row[0] <= cut_t]
        merged = sorted(prefix + shifted_suffix, key=lambda row: (row[0], row[1]))
        if not self._validate_candidate_actions(merged, log_on_error=True):
            self._log("suffix_replan_skip reason=merged_validation_failed")
            return actions

        old_makespan = max((t for t, _, _, _, _ in actions), default=-1)
        new_makespan = max((t for t, _, _, _, _ in merged), default=-1)
        if new_makespan < old_makespan:
            self._log(
                f"suffix_replan_accept robot={robot_id} sku={sku} streak={streak_len} "
                f"cut_t={cut_t} old={old_makespan} new={new_makespan}"
            )
            return merged

        self._log(
            f"suffix_replan_reject robot={robot_id} sku={sku} streak={streak_len} "
            f"cut_t={cut_t} old={old_makespan} new={new_makespan}"
        )
        return actions

    def _build_suffix_role_plans(self, robot_id: int, sku: int) -> Dict[int, List[str]]:
        plans: Dict[int, List[str]] = {
            rid: list(roles) for rid, roles in self.role_plans_by_robot.items()
        }
        token = f"dock_pallet_{sku}"
        existing = list(plans.get(robot_id, []))
        if not existing:
            existing = [ROLE_LOOP, ROLE_DELIVER_EASY, ROLE_DELIVER_HARD]
        if token in existing:
            plans[robot_id] = existing
        else:
            plans[robot_id] = [token] + existing
        return plans

    def _build_suffix_state(
        self,
        *,
        robot_positions: List[Tuple[int, int]],
        pallet_map: Dict[Tuple[int, int], int],
        remaining_orders: List[collections.Counter],
    ):
        width = self.state.width
        height = self.state.height
        grid = np.zeros((height, width), dtype=int)
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        for (px, py) in pallet_map.keys():
            grid[py, px] = 2
        for rx, ry in robot_positions:
            grid[ry, rx] = 3

        return SimpleNamespace(
            width=width,
            height=height,
            grid=grid,
            robots=list(robot_positions),
            pallets=dict(pallet_map),
            orders=[collections.Counter(order) for order in remaining_orders],
        )

    def _find_hot_sku_streak_candidate(
        self,
        actions: List[Tuple[int, int, str, int, int]],
        threshold: int,
    ) -> Tuple[int, int, int, int] | None:
        events_by_robot = self._extract_fulfill_events(actions)
        best: Tuple[int, int, int, int] | None = None
        all_skus = list(self.skus_by_demand)
        if not all_skus:
            all_skus = sorted({sku for order in self.orders for sku in order.items.keys()})

        for rid, events in events_by_robot.items():
            if len(events) < threshold:
                continue
            for sku in all_skus:
                run = 0
                run_start_t = -1
                for event in events:
                    if sku in event.skus:
                        if run == 0:
                            run_start_t = event.timestep
                        run += 1
                        if run >= threshold:
                            candidate = (rid, sku, run, run_start_t)
                            if best is None or (candidate[2], -candidate[3], -candidate[0], -candidate[1]) > (
                                best[2],
                                -best[3],
                                -best[0],
                                -best[1],
                            ):
                                best = candidate
                    else:
                        run = 0
                        run_start_t = -1
        return best

    def _extract_fulfill_events(
        self,
        actions: List[Tuple[int, int, str, int, int]],
    ) -> Dict[int, List[RobotFulfillEvent]]:
        by_t: Dict[int, List[Tuple[int, str, int, int]]] = collections.defaultdict(list)
        max_t = -1
        for t, rid, action, x, y in actions:
            by_t[t].append((rid, action.lower(), x, y))
            if t > max_t:
                max_t = t

        robot_pos: List[List[int]] = [[x, y] for x, y in self.state.robots]
        robot_storage: List[collections.Counter] = [collections.Counter() for _ in self.state.robots]
        robot_docked: List[List[int]] = [[] for _ in self.state.robots]
        pallets: List[dict] = []
        pallet_at: Dict[int, int] = {}
        for pid, ((x, y), sku) in enumerate(self.state.pallets.items()):
            pallets.append({"id": pid, "x": x, "y": y, "sku": sku, "docked_to": None})
            pallet_at[100 * y + x] = pid

        order_defs = [collections.Counter(order.items) for order in self.orders]
        orders_fulfilled = [False for _ in order_defs]
        events_by_robot: Dict[int, List[RobotFulfillEvent]] = collections.defaultdict(list)

        for t in range(max_t + 1):
            timestep_actions = by_t.get(t, [])

            for rid, action, x, y in timestep_actions:
                if action != "undock":
                    continue
                pid = pallet_at.get(100 * y + x)
                if pid is None:
                    continue
                if pallets[pid]["docked_to"] != rid:
                    continue
                pallets[pid]["docked_to"] = None
                robot_docked[rid] = [p for p in robot_docked[rid] if p != pid]

            for rid, action, x, y in timestep_actions:
                if action != "pick":
                    continue
                pid = pallet_at.get(100 * y + x)
                if pid is None:
                    continue
                sku = pallets[pid]["sku"]
                robot_storage[rid][sku] += 1

            for rid, action, x, y in timestep_actions:
                if action != "dock":
                    continue
                pid = pallet_at.get(100 * y + x)
                if pid is None:
                    continue
                if pallets[pid]["docked_to"] is not None:
                    continue
                pallets[pid]["docked_to"] = rid
                if pid not in robot_docked[rid]:
                    robot_docked[rid].append(pid)

            for rid, action, x, y in timestep_actions:
                if action != "move":
                    continue
                old_x, old_y = robot_pos[rid]
                dx = x - old_x
                dy = y - old_y
                robot_pos[rid][0] = x
                robot_pos[rid][1] = y
                for pid in robot_docked[rid]:
                    pallet = pallets[pid]
                    old_key = 100 * pallet["y"] + pallet["x"]
                    pallet_at.pop(old_key, None)
                    pallet["x"] += dx
                    pallet["y"] += dy
                    pallet_at[100 * pallet["y"] + pallet["x"]] = pid

            for rid, action, _, _ in timestep_actions:
                if action != "fulfill":
                    continue
                bag = collections.Counter(robot_storage[rid])
                matched_order_id = None
                for oid, order_bag in enumerate(order_defs):
                    if orders_fulfilled[oid]:
                        continue
                    if order_bag == bag:
                        orders_fulfilled[oid] = True
                        matched_order_id = oid
                        break
                events_by_robot[rid].append(
                    RobotFulfillEvent(
                        timestep=t,
                        robot_id=rid,
                        order_id=matched_order_id,
                        skus=set(bag.keys()),
                    )
                )
                robot_storage[rid].clear()

        return events_by_robot

    def _latest_clean_snapshot_before(
        self,
        actions: List[Tuple[int, int, str, int, int]],
        target_t: int,
    ) -> Tuple[int, List[Tuple[int, int]], Dict[Tuple[int, int], int], set[int]] | None:
        by_t: Dict[int, List[Tuple[int, str, int, int]]] = collections.defaultdict(list)
        max_t = -1
        for t, rid, action, x, y in actions:
            by_t[t].append((rid, action.lower(), x, y))
            if t > max_t:
                max_t = t

        robot_pos: List[List[int]] = [[x, y] for x, y in self.state.robots]
        robot_storage: List[collections.Counter] = [collections.Counter() for _ in self.state.robots]
        robot_docked: List[List[int]] = [[] for _ in self.state.robots]
        pallets: List[dict] = []
        pallet_at: Dict[int, int] = {}
        for pid, ((x, y), sku) in enumerate(self.state.pallets.items()):
            pallets.append({"id": pid, "x": x, "y": y, "sku": sku, "docked_to": None})
            pallet_at[100 * y + x] = pid

        order_defs = [collections.Counter(order.items) for order in self.orders]
        orders_fulfilled = [False for _ in order_defs]

        latest = (
            -1,
            [tuple(pos) for pos in robot_pos],
            {((x, y)): sku for (x, y), sku in self.state.pallets.items()},
            set(),
        )
        horizon = min(max_t, target_t)
        for t in range(horizon + 1):
            timestep_actions = by_t.get(t, [])

            for rid, action, x, y in timestep_actions:
                if action != "undock":
                    continue
                pid = pallet_at.get(100 * y + x)
                if pid is None:
                    continue
                if pallets[pid]["docked_to"] != rid:
                    continue
                pallets[pid]["docked_to"] = None
                robot_docked[rid] = [p for p in robot_docked[rid] if p != pid]

            for rid, action, x, y in timestep_actions:
                if action != "pick":
                    continue
                pid = pallet_at.get(100 * y + x)
                if pid is None:
                    continue
                sku = pallets[pid]["sku"]
                robot_storage[rid][sku] += 1

            for rid, action, x, y in timestep_actions:
                if action != "dock":
                    continue
                pid = pallet_at.get(100 * y + x)
                if pid is None:
                    continue
                if pallets[pid]["docked_to"] is not None:
                    continue
                pallets[pid]["docked_to"] = rid
                if pid not in robot_docked[rid]:
                    robot_docked[rid].append(pid)

            for rid, action, x, y in timestep_actions:
                if action != "move":
                    continue
                old_x, old_y = robot_pos[rid]
                dx = x - old_x
                dy = y - old_y
                robot_pos[rid][0] = x
                robot_pos[rid][1] = y
                for pid in robot_docked[rid]:
                    pallet = pallets[pid]
                    old_key = 100 * pallet["y"] + pallet["x"]
                    pallet_at.pop(old_key, None)
                    pallet["x"] += dx
                    pallet["y"] += dy
                    pallet_at[100 * pallet["y"] + pallet["x"]] = pid

            for rid, action, _, _ in timestep_actions:
                if action != "fulfill":
                    continue
                bag = collections.Counter(robot_storage[rid])
                for oid, order_bag in enumerate(order_defs):
                    if orders_fulfilled[oid]:
                        continue
                    if order_bag == bag:
                        orders_fulfilled[oid] = True
                        break
                robot_storage[rid].clear()

            is_clean = all(not storage for storage in robot_storage) and all(
                not docked for docked in robot_docked
            )
            if is_clean:
                pallet_map = {
                    (pallet["x"], pallet["y"]): int(pallet["sku"])
                    for pallet in pallets
                    if pallet["docked_to"] is None
                }
                latest = (
                    t,
                    [tuple(pos) for pos in robot_pos],
                    pallet_map,
                    {i for i, done in enumerate(orders_fulfilled) if done},
                )

        if latest[0] < 0:
            return None
        return latest

    def _log_solve_start(self, total_orders: int) -> None:
        self._log(f"solve_start total_orders={total_orders}")
        if self.relocation_plan:
            summary = ", ".join(
                f"SKU{job.sku}->{job.bucket}@{job.placement_offset}(score={job.score:.2f})"
                for job in list(self.relocation_plan)[:6]
            )
            self._log(f"relocation_plan count={len(self.relocation_plan)} top=[{summary}]")
        else:
            self._log("relocation_plan count=0")

    def _execute_dispatch(
        self,
        *,
        robot: RobotState,
        remaining_orders: Deque[int],
        completed: int,
        total_orders: int,
        dispatch_number: int,
    ) -> bool:
        """
        Pattern: command-dispatch template.
        Select role token, execute role with fallbacks, and emit consistent lifecycle logs.
        """
        role_token, from_plan = self._dispatch_role(robot)
        role, strategy, forced_reloc_sku = self._decode_role_token(role_token, robot.id)
        if role == ROLE_DELIVER:
            self._ensure_warmup_barrier()

        if robot.last_t < self._dispatch_floor_t:
            robot.last_t = self._dispatch_floor_t

        self._log_dispatch_start(
            dispatch_number=dispatch_number,
            robot=robot,
            role_token=role_token,
            completed=completed,
            total_orders=total_orders,
            remaining=len(remaining_orders),
        )
        t0 = time.perf_counter()
        handled = self._execute_role_with_fallbacks(
            robot=robot,
            role=role,
            strategy=strategy,
            forced_reloc_sku=forced_reloc_sku,
            dispatch_number=dispatch_number,
            from_plan=from_plan,
            remaining_orders=remaining_orders,
        )
        elapsed = time.perf_counter() - t0
        self._log_dispatch_end(
            dispatch_number=dispatch_number,
            robot=robot,
            role=role,
            handled=handled,
            elapsed_s=elapsed,
            completed=total_orders - len(remaining_orders),
            total_orders=total_orders,
            remaining=len(remaining_orders),
        )
        return handled

    def _execute_role_with_fallbacks(
        self,
        *,
        robot: RobotState,
        role: str,
        strategy: str,
        forced_reloc_sku: int | None,
        dispatch_number: int,
        from_plan: bool,
        remaining_orders: Deque[int],
    ) -> bool:
        if role == ROLE_DELIVER:
            handled = self._role_deliver(robot, remaining_orders, strategy=strategy)
            if handled and not from_plan:
                self._toggle_delivery_strategy(robot.id)
            return handled

        handled = self._role_relocate_pallet(
            robot,
            remaining_orders,
            forced_sku=forced_reloc_sku,
            dispatch_number=dispatch_number,
        )
        if not handled:
            self._ensure_warmup_barrier()
            handled = self._deliver_with_robot_strategy(robot, remaining_orders)
        if not handled:
            handled = self._fallback_deliver_any_robot(remaining_orders)
        return handled

    def _log_dispatch_start(
        self,
        *,
        dispatch_number: int,
        robot: RobotState,
        role_token: str,
        completed: int,
        total_orders: int,
        remaining: int,
    ) -> None:
        if dispatch_number % self.config.dispatch_log_every != 0:
            return
        self._log(
            f"dispatch_start n={dispatch_number} robot={robot.id} role={role_token} "
            f"robot_t={robot.last_t} completed={completed}/{total_orders} remaining={remaining}"
        )

    def _log_dispatch_end(
        self,
        *,
        dispatch_number: int,
        robot: RobotState,
        role: str,
        handled: bool,
        elapsed_s: float,
        completed: int,
        total_orders: int,
        remaining: int,
    ) -> None:
        if dispatch_number % self.config.dispatch_log_every != 0:
            return
        self._log(
            f"dispatch_end n={dispatch_number} robot={robot.id} role={role} success={handled} "
            f"elapsed_s={elapsed_s:.2f} completed={completed}/{total_orders} remaining={remaining}"
        )

    def _maybe_log_progress(self, *, completed: int, total_orders: int, dispatch_count: int) -> None:
        if completed % self.config.progress_every != 0:
            return
        current_makespan = max(r.last_t for r in self.robots)
        print(
            f"[solver] planned {completed}/{total_orders} orders, "
            f"current makespan={current_makespan}, dispatches={dispatch_count}"
        )
        self._log(
            f"progress completed={completed}/{total_orders} "
            f"makespan={current_makespan} dispatches={dispatch_count}"
        )

    def _elapsed_plan_seconds(self) -> float:
        if self._plan_started_monotonic <= 0:
            return 0.0
        return time.monotonic() - self._plan_started_monotonic

    def _check_global_limits_or_raise(self, remaining_orders: Deque[int]) -> None:
        if self.config.max_plan_time_seconds > 0:
            elapsed = self._elapsed_plan_seconds()
            if elapsed >= self.config.max_plan_time_seconds:
                raise TimeoutError(
                    "Reached max_plan_time_seconds "
                    f"({self.config.max_plan_time_seconds:.1f}s) with {len(remaining_orders)} orders remaining."
                )

        if self.config.max_makespan is not None and self.config.max_makespan >= 0:
            current_makespan = max((r.last_t for r in self.robots), default=-1)
            if current_makespan > self.config.max_makespan:
                raise RuntimeError(
                    f"Reached max_makespan={self.config.max_makespan} at makespan={current_makespan} "
                    f"with {len(remaining_orders)} orders remaining."
                )

    def _build_ranked_order_queue(self) -> Deque[int]:
        optimizer = OrderOptimizer(self.state.pallets)
        scored = optimizer.sort_orders_by_cluster_efficiency([o.items for o in self.orders])
        return collections.deque(item["order_idx"] for item in scored)

    def _recalculate_order_costs(self, order_ids: Deque[int] | List[int] | None = None) -> None:
        if order_ids is None:
            ids = range(len(self.orders))
        else:
            ids = list(order_ids)
        for order_idx in ids:
            self.orders[order_idx].estimate_cost(self.edge_scorer)

    def _toggle_delivery_strategy(self, robot_id: int) -> None:
        current = self._next_delivery_strategy_by_robot.get(robot_id, DELIVER_EASY)
        self._next_delivery_strategy_by_robot[robot_id] = (
            DELIVER_HARD if current == DELIVER_EASY else DELIVER_EASY
        )

    def _deliver_with_robot_strategy(self, robot: RobotState, remaining_orders: Deque[int]) -> bool:
        strategy = self._next_delivery_strategy_by_robot.get(robot.id, DELIVER_EASY)
        handled = self._role_deliver(robot, remaining_orders, strategy=strategy)
        if handled:
            self._toggle_delivery_strategy(robot.id)
        return handled

    def _normalize_role_plans(
        self, role_plans: Dict[int, List[str]]
    ) -> Dict[int, List[str]]:
        normalized: Dict[int, List[str]] = {}
        allowed = {
            ROLE_RELOCATE_PALLET,
            ROLE_DELIVER,
            ROLE_DELIVER_EASY,
            ROLE_DELIVER_HARD,
            ROLE_LOOP,
        }
        for robot_id, roles in role_plans.items():
            rid = int(robot_id)
            cleaned = [str(role).strip().lower() for role in roles if str(role).strip()]
            if not cleaned:
                continue
            for role in cleaned:
                if role not in allowed and self._parse_forced_relocate_role(role) is None:
                    raise ValueError(
                        f"Unknown role '{role}' in plan for robot {rid}. "
                        f"Allowed: {sorted(allowed)}"
                    )
            normalized[rid] = cleaned
        return normalized

    def _parse_forced_relocate_role(self, role_token: str) -> int | None:
        token = role_token.strip().lower()
        prefix = "dock_pallet_"
        if not token.startswith(prefix):
            return None
        sku_part = token[len(prefix) :]
        if not sku_part.isdigit():
            return None
        return int(sku_part)

    def _build_relocation_plan(self) -> Deque[RelocationJob]:
        """
        Algorithm: weighted lift heuristic over bucket-level SKU frequencies.
        Pattern: plan builder that ranks relocation candidates then decorates them with targets.
        """
        forced_skus = list(dict.fromkeys(self.config.relocation_skus_to_relocate or []))
        metadata_db_path = self._metadata_db_path
        bucket_items: Dict[str, int] = {}
        bucket_sku_counts: Dict[str, collections.Counter] = {}
        if metadata_db_path is not None:
            metadata_db_path = Path(metadata_db_path)
            if metadata_db_path.exists():
                bucket_items, bucket_sku_counts = self._load_bucket_stats_from_metadata_db(
                    metadata_db_path
                )

        if forced_skus:
            forced_jobs = self._build_forced_relocation_jobs(forced_skus, bucket_sku_counts)
            self._assign_relocation_targets(forced_jobs)
            return collections.deque(forced_jobs)

        tracked_buckets = [b for b in BUCKET_TO_HOTSPOT.keys() if b in bucket_items]
        total_items = sum(bucket_items.get(b, 0) for b in tracked_buckets)
        if total_items <= 0:
            return collections.deque()

        sku_totals: collections.Counter = collections.Counter()
        for bucket in tracked_buckets:
            sku_totals.update(bucket_sku_counts.get(bucket, collections.Counter()))

        jobs: List[RelocationJob] = []
        for sku, sku_total in sku_totals.items():
            if sku_total <= 0:
                continue
            if not self.scheduler.has_sku(sku):
                continue

            best = None
            for bucket in tracked_buckets:
                bucket_sku = bucket_sku_counts.get(bucket, collections.Counter())
                count = bucket_sku.get(sku, 0)
                if count <= 0:
                    continue

                sku_share = count / sku_total
                bucket_share = bucket_items[bucket] / total_items
                lift = sku_share - bucket_share
                if lift < self.config.relocation_min_lift:
                    continue

                score = lift * sku_total
                candidate = (score, lift, bucket, count)
                if best is None or candidate > best:
                    best = candidate

            if best is None:
                continue

            score, lift, bucket, count = best
            hotspot = BUCKET_TO_HOTSPOT[bucket]
            sku_cells = self._metadata_sku_cells.get(sku, [])
            if sku_cells:
                hotspot = (sku_cells[0][0], sku_cells[0][1])
            jobs.append(
                RelocationJob(
                    sku=sku,
                    bucket=bucket,
                    hotspot=hotspot,
                    score=float(score),
                    metadata={
                        "lift": float(lift),
                        "sku_total": float(sku_total),
                        "bucket_count": float(count),
                    },
                )
            )

        jobs.sort(key=lambda j: (-j.score, j.sku))
        jobs = jobs[: self.config.relocation_top_skus]
        self._assign_relocation_targets(jobs)
        return collections.deque(jobs)

    def _build_forced_relocation_jobs(
        self,
        forced_skus: List[int],
        bucket_sku_counts: Dict[str, collections.Counter],
    ) -> List[RelocationJob]:
        jobs: List[RelocationJob] = []
        for idx, sku in enumerate(forced_skus):
            if not self.scheduler.has_sku(sku):
                continue
            bucket = self._choose_bucket_for_sku(sku, bucket_sku_counts)
            hotspot = BUCKET_TO_HOTSPOT[bucket]
            sku_cells = self._metadata_sku_cells.get(sku, [])
            if sku_cells:
                hotspot = (sku_cells[0][0], sku_cells[0][1])
            jobs.append(
                RelocationJob(
                    sku=sku,
                    bucket=bucket,
                    hotspot=hotspot,
                    score=float(1_000_000 - idx),
                    metadata={"forced": 1.0, "forced_rank": float(idx)},
                )
            )
        return jobs

    def _choose_bucket_for_sku(
        self,
        sku: int,
        bucket_sku_counts: Dict[str, collections.Counter],
    ) -> str:
        best_bucket = None
        best_count = 0
        for bucket in BUCKET_TO_HOTSPOT.keys():
            count = int(bucket_sku_counts.get(bucket, collections.Counter()).get(sku, 0))
            if count > best_count:
                best_count = count
                best_bucket = bucket
        if best_bucket is not None and best_count > 0:
            return best_bucket

        pallet_cells = self.scheduler.pallet_cells_for_sku(sku)
        if pallet_cells:
            px, py = pallet_cells[0]
            return self._closest_bucket_for_cell(px, py)
        return "left_edge"

    def _assign_relocation_targets(self, jobs: List[RelocationJob]) -> None:
        planned_targets: set[Tuple[int, int]] = set()
        for job in jobs:
            target_xy = self._choose_metadata_guided_relocation_target(
                job=job,
                reserved_targets=planned_targets,
            )
            if target_xy is not None:
                tx, ty = target_xy
                job.preferred_target_xy = target_xy
                job.placement_offset = (tx - job.hotspot[0], ty - job.hotspot[1])
                planned_targets.add(target_xy)
                continue

            offset = self._choose_unique_relocation_offset(
                bucket=job.bucket,
                hotspot=job.hotspot,
                reserved_targets=planned_targets,
            )
            job.placement_offset = offset
            tx, ty = job.hotspot[0] + offset[0], job.hotspot[1] + offset[1]
            if 0 <= tx < self.state.width and 0 <= ty < self.state.height:
                job.preferred_target_xy = (tx, ty)
                planned_targets.add((tx, ty))

    def _find_default_metadata_db_path(self) -> Path | None:
        path = Path("output") / "solution_metadata.db"
        if path.exists():
            return path
        return None

    def _bucket_for_edge(self, x: int, y: int) -> str:
        if x == 0:
            return "left_edge"
        if x == self.state.width - 1:
            return "right_edge"
        if y == 0:
            return "top_x0_29" if x < (self.state.width // 2) else "top_x30_59"
        if y == self.state.height - 1:
            return "bottom_x0_29" if x < (self.state.width // 2) else "bottom_x30_59"
        return "non_edge"

    def _load_bucket_stats_from_metadata_db(
        self, metadata_db_path: Path
    ) -> Tuple[Dict[str, int], Dict[str, collections.Counter]]:
        bucket_items: Dict[str, int] = {}
        bucket_sku_counts: Dict[str, collections.Counter] = {}

        conn = sqlite3.connect(metadata_db_path)
        try:
            try:
                run_id = self._select_best_non_test_run_id(conn)
                if run_id is None:
                    return bucket_items, bucket_sku_counts
                rows = conn.execute(
                    "SELECT x, y, skus_json FROM fulfills WHERE run_id = ?",
                    (run_id,),
                ).fetchall()
            except sqlite3.Error:
                return bucket_items, bucket_sku_counts
        finally:
            conn.close()

        for x, y, skus_json in rows:
            bucket = self._bucket_for_edge(int(x), int(y))
            bucket_counter = bucket_sku_counts.setdefault(bucket, collections.Counter())
            try:
                sku_values = json.loads(skus_json)
            except Exception:
                sku_values = []
            if not isinstance(sku_values, list):
                continue
            for sku_raw in sku_values:
                try:
                    sku = int(sku_raw)
                except (TypeError, ValueError):
                    continue
                bucket_counter[sku] += 1
                bucket_items[bucket] = bucket_items.get(bucket, 0) + 1

        return bucket_items, bucket_sku_counts

    def _is_relocation_target_cell_allowed(
        self,
        cell: Tuple[int, int],
        *,
        reserved_targets: set[Tuple[int, int]],
        block_high_use: bool = True,
    ) -> bool:
        x, y = cell
        if not (0 <= x < self.state.width and 0 <= y < self.state.height):
            return False
        if cell in self.scheduler.pallets:
            return False
        if cell in reserved_targets:
            return False
        if cell in self.travel_lane_cells:
            return False
        if block_high_use and cell in self._metadata_high_use_cells:
            return False
        return True

    def _iter_manhattan_cells(
        self, center_x: int, center_y: int, *, max_radius: int
    ):
        for radius in range(0, max_radius + 1):
            min_x = max(0, center_x - radius)
            max_x = min(self.state.width - 1, center_x + radius)
            min_y = max(0, center_y - radius)
            max_y = min(self.state.height - 1, center_y + radius)
            for tx in range(min_x, max_x + 1):
                for ty in range(min_y, max_y + 1):
                    if abs(tx - center_x) + abs(ty - center_y) != radius:
                        continue
                    yield tx, ty, radius

    def _iter_sku_anchor_rows(self, sku: int, *, limit: int) -> List[Tuple[int, int, int]]:
        sku_rows = self._metadata_sku_cells.get(sku, [])
        if not sku_rows:
            return []
        return sku_rows[: min(len(sku_rows), limit)]

    def _choose_metadata_guided_relocation_target(
        self,
        *,
        job: RelocationJob,
        reserved_targets: set[Tuple[int, int]],
    ) -> Tuple[int, int] | None:
        """
        Algorithm: constrained nearest-neighbor search on Manhattan rings.
        Optimizes for low-use cells near high SKU-flow anchors under lane/occupancy constraints.
        """
        anchor_rows = self._iter_sku_anchor_rows(job.sku, limit=24)
        if not anchor_rows:
            return None

        best: Tuple[int, int] | None = None
        best_key: Tuple[int, int, int, int] | None = None
        for sx, sy, sku_count in anchor_rows:
            for tx, ty, radius in self._iter_manhattan_cells(sx, sy, max_radius=9):
                cell = (tx, ty)
                if not self._is_relocation_target_cell_allowed(
                    cell,
                    reserved_targets=reserved_targets,
                    block_high_use=True,
                ):
                    continue
                use_score = self._metadata_use_by_cell.get(cell, 0)
                key = (
                    use_score,
                    radius,
                    abs(tx - job.hotspot[0]) + abs(ty - job.hotspot[1]),
                    -sku_count,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = cell

        if best is not None:
            return best

        # If no strict low-use option exists, retry allowing high-use (still avoids lanes).
        for sx, sy, _ in anchor_rows:
            for tx, ty, _ in self._iter_manhattan_cells(sx, sy, max_radius=11):
                cell = (tx, ty)
                if self._is_relocation_target_cell_allowed(
                    cell,
                    reserved_targets=reserved_targets,
                    block_high_use=False,
                ):
                    return cell
        return None

    def _metadata_candidate_targets_for_sku(self, sku: int, limit: int = 64) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        seen: set[Tuple[int, int]] = set()
        anchor_rows = self._iter_sku_anchor_rows(sku, limit=18)
        if not anchor_rows:
            return out

        for sx, sy, _ in anchor_rows:
            for tx, ty, _ in self._iter_manhattan_cells(sx, sy, max_radius=4):
                cell = (tx, ty)
                if cell in seen:
                    continue
                if cell in self.scheduler.pallets:
                    continue
                if cell in self.travel_lane_cells:
                    continue
                seen.add(cell)
                out.append(cell)
                if len(out) >= limit:
                    return out
        return out

    def _next_available_robot(self) -> RobotState:
        return min(self.robots, key=lambda r: (r.last_t, r.id))

    def _ensure_warmup_barrier(self) -> None:
        if self._warmup_barrier_applied:
            return
        if self._initial_relocate_assigned <= 0:
            return
        if not self.robots:
            return
        floor_t = max(r.last_t for r in self.robots)
        if floor_t > self._dispatch_floor_t:
            self._dispatch_floor_t = floor_t
        self._warmup_barrier_applied = True
        self._log(f"warmup_barrier floor_t={self._dispatch_floor_t}")

    def _dispatch_role(self, robot: RobotState) -> Tuple[str, bool]:
        """
        Pattern: strategy selection (policy-based dispatch).
        Chooses between planned-role strategy, relocation strategy, or delivery strategy.
        """
        planned = self._next_role_from_plan(robot.id)
        if planned is not None:
            return planned, True

        if (
            self._initial_relocate_assigned < self.config.initial_relocate_dispatches
            and self._has_relocation_candidate()
        ):
            self._initial_relocate_assigned += 1
            return ROLE_RELOCATE_PALLET, False

        if self._has_relocation_candidate():
            if self.rng.random() < self.config.relocate_pallet_probability:
                return ROLE_RELOCATE_PALLET, False

        strategy = self._next_delivery_strategy_by_robot.get(robot.id, DELIVER_EASY)
        if strategy == DELIVER_HARD:
            return ROLE_DELIVER_HARD, False
        return ROLE_DELIVER_EASY, False

    def _next_role_from_plan(self, robot_id: int) -> str | None:
        cursor = self._role_cursors_by_robot.get(robot_id)
        if cursor is None or not cursor.roles:
            return None

        max_steps = len(cursor.roles) + 2
        steps = 0
        while steps < max_steps:
            if cursor.next_index >= len(cursor.roles):
                if cursor.loop_index is not None:
                    cursor.next_index = cursor.loop_index
                else:
                    cursor.next_index = 0

            token = cursor.roles[cursor.next_index]
            cursor.next_index += 1
            if token == ROLE_LOOP:
                steps += 1
                continue
            return token

        raise RuntimeError(f"Robot {robot_id} plan does not contain any dispatchable roles.")

    def _decode_role_token(self, role_token: str, robot_id: int) -> Tuple[str, str, int | None]:
        token = role_token.strip().lower()
        forced_sku = self._parse_forced_relocate_role(token)
        if forced_sku is not None:
            return ROLE_RELOCATE_PALLET, DELIVER_EASY, forced_sku
        if token == ROLE_RELOCATE_PALLET:
            return ROLE_RELOCATE_PALLET, DELIVER_EASY, None
        if token == ROLE_DELIVER:
            return ROLE_DELIVER, DELIVER_EASY, None
        if token == ROLE_DELIVER_EASY:
            return ROLE_DELIVER, DELIVER_EASY, None
        if token == ROLE_DELIVER_HARD:
            return ROLE_DELIVER, DELIVER_HARD, None
        raise ValueError(f"Unknown role token '{role_token}' for robot {robot_id}")

    def _has_relocation_candidate(self) -> bool:
        while self.relocation_plan:
            head = self.relocation_plan[0]
            if head.sku in self.relocated_skus:
                self.relocation_plan.popleft()
                continue
            if not self.scheduler.has_sku(head.sku):
                self.relocation_plan.popleft()
                continue
            return True
        return False

    def _inject_lookahead_relocation_jobs(self, remaining_orders: Deque[int]) -> None:
        threshold = int(self.config.min_jobs_for_dock)
        if threshold <= 1:
            return
        if len(remaining_orders) < threshold:
            return

        lookahead_ids = list(remaining_orders)[:threshold]
        sku_job_hits: collections.Counter = collections.Counter()
        for order_idx in lookahead_ids:
            for sku in self.orders[order_idx].items.keys():
                sku_job_hits[sku] += 1

        candidates = [
            (hits, sku)
            for sku, hits in sku_job_hits.items()
            if hits >= threshold
            and sku not in self.relocated_skus
            and sku not in self._lookahead_relocation_seeded_skus
            and self.scheduler.has_sku(sku)
        ]
        if not candidates:
            return

        candidates.sort(key=lambda row: (-row[0], row[1]))
        seeded = 0
        for hits, sku in candidates:
            if seeded >= 3:
                break
            job = self._build_lookahead_relocation_job(sku=sku, hits=hits)
            if job is None:
                continue
            self.relocation_plan.appendleft(job)
            self._lookahead_relocation_seeded_skus.add(sku)
            seeded += 1
            self._log(
                f"lookahead_relocation_seed sku={sku} hits={hits} "
                f"bucket={job.bucket} target={job.preferred_target_xy} score={job.score:.2f}"
            )

    def _build_lookahead_relocation_job(self, *, sku: int, hits: int) -> RelocationJob | None:
        pallets = self.scheduler.pallet_cells_for_sku(sku)
        if not pallets:
            return None
        anchor_xy = pallets[0]
        sku_cells = self._metadata_sku_cells.get(sku, [])
        if sku_cells:
            anchor_xy = (sku_cells[0][0], sku_cells[0][1])
        bucket = self._closest_bucket_for_cell(anchor_xy[0], anchor_xy[1])
        hotspot = anchor_xy
        score = float(hits * 10)
        job = RelocationJob(sku=sku, bucket=bucket, hotspot=hotspot, score=score)

        reserved = set(self.relocated_pallet_targets)
        for queued in self.relocation_plan:
            if queued.preferred_target_xy is not None:
                reserved.add(queued.preferred_target_xy)
        target_xy = self._choose_metadata_guided_relocation_target(
            job=job,
            reserved_targets=reserved,
        )
        if target_xy is None:
            offset = self._choose_unique_relocation_offset(
                bucket=job.bucket,
                hotspot=job.hotspot,
                reserved_targets=reserved,
            )
            job.placement_offset = offset
            tx, ty = hotspot[0] + offset[0], hotspot[1] + offset[1]
            if 0 <= tx < self.state.width and 0 <= ty < self.state.height:
                job.preferred_target_xy = (tx, ty)
        else:
            tx, ty = target_xy
            job.preferred_target_xy = target_xy
            job.placement_offset = (tx - hotspot[0], ty - hotspot[1])
        return job

    def _closest_bucket_for_cell(self, x: int, y: int) -> str:
        choices = list(BUCKET_TO_HOTSPOT.items())
        _, bucket = min(
            ((abs(hx - x) + abs(hy - y), b) for b, (hx, hy) in choices),
            key=lambda row: (row[0], row[1]),
        )
        return bucket

    def _role_deliver(self, robot: RobotState, remaining_orders: Deque[int], strategy: str) -> bool:
        """
        Algorithm: greedy best-first attempt over a bounded candidate window.
        The ranking key switches by strategy ('easy' vs 'hard').
        """
        if not remaining_orders:
            return False
        if strategy not in {DELIVER_EASY, DELIVER_HARD}:
            raise ValueError(f"Unknown deliver strategy: {strategy}")

        candidate_window = min(len(remaining_orders), self.config.delivery_candidate_window)
        candidate_order_ids = list(remaining_orders)[:candidate_window]
        ranked_order_ids = sorted(
            candidate_order_ids,
            key=lambda oid: (
                self.orders[oid].estimated_cost if strategy == DELIVER_EASY else -self.orders[oid].estimated_cost,
                oid,
            ),
        )
        ranked_order_ids = ranked_order_ids[: self.config.max_delivery_order_attempts]

        for order_idx in ranked_order_ids:
            order = self.orders[order_idx].items
            if self._plan_order_for_robot(order_idx, order, robot):
                try:
                    remaining_orders.remove(order_idx)
                except ValueError:
                    pass
                return True
        return False

    def _fallback_deliver_any_robot(self, remaining_orders: Deque[int]) -> bool:
        for robot in sorted(self.robots, key=lambda r: (r.last_t, r.id)):
            if self._deliver_with_robot_strategy(robot, remaining_orders):
                return True
        return False

    def _role_relocate_pallet(
        self,
        robot: RobotState,
        remaining_orders: Deque[int],
        forced_sku: int | None = None,
        dispatch_number: int | None = None,
    ) -> bool:
        if forced_sku is not None:
            return self._relocate_forced_sku(
                robot,
                remaining_orders,
                forced_sku,
                dispatch_number=dispatch_number,
            )

        if not self._has_relocation_candidate():
            return False

        attempts = min(3, len(self.relocation_plan))
        for _ in range(attempts):
            if not self.relocation_plan:
                return False
            job = self.relocation_plan[0]
            if job.sku in self.relocated_skus:
                self.relocation_plan.popleft()
                continue

            ok = self._plan_relocate_pallet_for_robot(
                robot=robot,
                job=job,
            )
            if ok:
                self._recalculate_order_costs(remaining_orders)
                self.relocated_skus.add(job.sku)
                self.relocation_plan.popleft()
                self._log(
                    f"relocation_done sku={job.sku} bucket={job.bucket} "
                    f"hotspot={job.hotspot} offset={job.placement_offset} "
                    f"target={job.preferred_target_xy} score={job.score:.3f}"
                )
                return True

            job.attempts += 1
            self.relocation_plan.rotate(-1)
            if job.attempts >= self.config.relocation_max_attempts_per_sku:
                try:
                    self.relocation_plan.remove(job)
                except ValueError:
                    pass
        return False

    def _relocate_forced_sku(
        self,
        robot: RobotState,
        remaining_orders: Deque[int],
        forced_sku: int,
        dispatch_number: int | None = None,
    ) -> bool:
        if (
            dispatch_number is not None
            and not self._should_attempt_forced_dock(robot.id, forced_sku, dispatch_number)
        ):
            return False
        if forced_sku in self.relocated_skus:
            return False
        if not self.scheduler.has_sku(forced_sku):
            return False
        forced_job = self._build_lookahead_relocation_job(
            sku=forced_sku,
            hits=max(1, self.config.min_jobs_for_dock),
        )
        if forced_job is None:
            return False
        ok = self._plan_relocate_pallet_for_robot(robot=robot, job=forced_job)
        if not ok:
            if dispatch_number is not None:
                self._record_forced_dock_failure(robot.id, forced_sku, dispatch_number)
            return False
        self._recalculate_order_costs(remaining_orders)
        self.relocated_skus.add(forced_sku)
        self._forced_dock_failures[(robot.id, forced_sku)] = 0
        self._forced_dock_cooldown_until_dispatch.pop((robot.id, forced_sku), None)
        self._log(
            f"forced_relocation_done sku={forced_sku} bucket={forced_job.bucket} "
            f"target={forced_job.preferred_target_xy}"
        )
        return True

    def _should_attempt_forced_dock(self, robot_id: int, sku: int, dispatch_number: int) -> bool:
        key = (robot_id, sku)
        cooldown_until = self._forced_dock_cooldown_until_dispatch.get(key)
        if cooldown_until is None:
            return True
        return dispatch_number >= cooldown_until

    def _record_forced_dock_failure(self, robot_id: int, sku: int, dispatch_number: int) -> None:
        key = (robot_id, sku)
        self._forced_dock_failures[key] += 1
        failures = self._forced_dock_failures[key]
        max_attempts = max(1, int(self.config.forced_dock_max_attempts_per_robot_sku))
        if failures < max_attempts:
            return

        cooldown = max(1, int(self.config.forced_dock_cooldown_dispatches))
        self._forced_dock_failures[key] = 0
        self._forced_dock_cooldown_until_dispatch[key] = dispatch_number + cooldown
        self._log(
            f"forced_dock_cooldown robot={robot_id} sku={sku} "
            f"until_dispatch={dispatch_number + cooldown}"
        )

    def _plan_order_for_robot(
        self, order_idx: int, order: collections.Counter, robot: RobotState
    ) -> bool:
        temp_robot = self._clone_robot_state(robot)
        remaining = collections.Counter(order)
        pending_actions: List[Tuple[int, int, str, int, int]] = []
        pending_paths: List[Tuple[RobotState, List[Tuple[int, int, int]]]] = []
        pending_footprints: List[Tuple[RobotState, int, int, int]] = []

        while sum(remaining.values()) > 0:
            options = self.scheduler.candidate_pick_options(remaining, (temp_robot.x, temp_robot.y))
            selected = None
            for _, sku, pallet_xy, pick_cell_xy in options:
                target_x, target_y = pick_cell_xy
                path = self._safe_plan_path(temp_robot, target_x, target_y)

                if path or (temp_robot.x == target_x and temp_robot.y == target_y):
                    pick_t = (path[-1][0] + 1) if path else (temp_robot.last_t + 1)
                    if not self._is_pick_target_static_at_time(pallet_xy, pick_t):
                        continue
                    selected = (sku, pallet_xy, path, pick_t)
                    break

            if selected is None:
                return False

            sku, pallet_xy, path, pick_t = selected
            if path:
                pending_paths.append((self._clone_robot_state(temp_robot), path))
            pending_actions.extend(self._apply_moves_to_robot(temp_robot, path))

            if not self.planner.can_occupy(temp_robot, pick_t, temp_robot.x, temp_robot.y):
                return False
            pallet_x, pallet_y = pallet_xy
            pending_actions.append((pick_t, temp_robot.id, "pick", pallet_x, pallet_y))
            pending_footprints.append((self._clone_robot_state(temp_robot), pick_t, temp_robot.x, temp_robot.y))
            temp_robot.last_t = pick_t
            temp_robot.storage[sku] += 1

            remaining[sku] -= 1
            if remaining[sku] <= 0:
                del remaining[sku]

        fulfill_x, fulfill_y = self.scheduler.best_fulfill_cell(temp_robot.x, temp_robot.y)
        fulfill_path = self._safe_plan_path(temp_robot, fulfill_x, fulfill_y)
        if not fulfill_path and (temp_robot.x != fulfill_x or temp_robot.y != fulfill_y):
            return False

        if fulfill_path:
            pending_paths.append((self._clone_robot_state(temp_robot), fulfill_path))
        pending_actions.extend(self._apply_moves_to_robot(temp_robot, fulfill_path))

        fulfill_t = temp_robot.last_t + 1
        if not self.planner.can_occupy(temp_robot, fulfill_t, temp_robot.x, temp_robot.y):
            return False
        pending_actions.append((fulfill_t, temp_robot.id, "fulfill", fulfill_x, fulfill_y))
        pending_footprints.append((self._clone_robot_state(temp_robot), fulfill_t, temp_robot.x, temp_robot.y))
        temp_robot.last_t = fulfill_t
        temp_robot.storage.clear()

        self._commit_plan(
            robot=robot,
            temp_robot=temp_robot,
            pending_actions=pending_actions,
            pending_paths=pending_paths,
            pending_footprints=pending_footprints,
        )
        return True

    def _select_relocation_source_pallet(
        self, robot: RobotState, sku: int
    ) -> Tuple[Tuple[int, int], int] | None:
        pallet_cells = self.scheduler.pallet_cells_for_sku(sku)
        if not pallet_cells:
            return None
        rx, ry = robot.x, robot.y
        pallet_xy = min(pallet_cells, key=lambda p: abs(p[0] - rx) + abs(p[1] - ry))
        pallet_id = self.pallet_id_by_coord.get(pallet_xy)
        if pallet_id is None:
            return None
        return pallet_xy, pallet_id

    def _candidate_relocation_stand_cells(
        self, robot: RobotState, pallet_xy: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        stand_cells = self.scheduler.pick_cells_for_pallet(pallet_xy)
        if not stand_cells:
            return []
        rx, ry = robot.x, robot.y
        stand_cells.sort(key=lambda p: abs(p[0] - rx) + abs(p[1] - ry))
        return stand_cells[: self.config.relocate_stand_candidate_limit]

    def _ranked_relocation_target_cells(
        self, job: RelocationJob, pallet_xy: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        target_pallet_cells = self._candidate_relocation_targets(job)
        target_pallet_cells.sort(
            key=lambda p: (
                -self._score_relocation_target(
                    old_xy=pallet_xy,
                    target_xy=p,
                    hotspot=job.hotspot,
                ),
                0 if p == job.preferred_target_xy else 1,
                abs(p[0] - pallet_xy[0]) + abs(p[1] - pallet_xy[1]),
            )
        )
        return target_pallet_cells[: self.config.relocate_target_candidate_limit]

    def _attempt_relocation_via_stand(
        self,
        *,
        robot: RobotState,
        pallet_xy: Tuple[int, int],
        pallet_id: int,
        stand_xy: Tuple[int, int],
        target_pallet_cells: List[Tuple[int, int]],
    ) -> Tuple[Tuple[int, int], int, int] | None:
        stand_x, stand_y = stand_xy
        temp_robot = self._clone_robot_state(robot)
        pending_actions: List[Tuple[int, int, str, int, int]] = []
        pending_paths: List[Tuple[RobotState, List[Tuple[int, int, int]]]] = []
        pending_footprints: List[Tuple[RobotState, int, int, int]] = []
        pending_static_additions: List[Tuple[int, int, int]] = []

        path_to_stand = self._safe_plan_path(temp_robot, stand_x, stand_y)
        if not path_to_stand and (temp_robot.x != stand_x or temp_robot.y != stand_y):
            return None
        if path_to_stand:
            pending_paths.append((self._clone_robot_state(temp_robot), path_to_stand))
        pending_actions.extend(self._apply_moves_to_robot(temp_robot, path_to_stand))

        dx = pallet_xy[0] - temp_robot.x
        dy = pallet_xy[1] - temp_robot.y
        if abs(dx) + abs(dy) != 1:
            return None

        dock_t = temp_robot.last_t + 1
        if not self.planner.can_occupy(temp_robot, dock_t, temp_robot.x, temp_robot.y):
            return None
        pending_actions.append((dock_t, temp_robot.id, "dock", pallet_xy[0], pallet_xy[1]))
        pending_footprints.append((self._clone_robot_state(temp_robot), dock_t, temp_robot.x, temp_robot.y))
        temp_robot.last_t = dock_t
        temp_robot.docks[(dx, dy)] = pallet_id

        chosen_target: Tuple[int, int, List[Tuple[int, int, int]]] | None = None
        for tx, ty in target_pallet_cells:
            if (tx, ty) in self.scheduler.pallets and (tx, ty) != pallet_xy:
                continue

            target_robot_x = tx - dx
            target_robot_y = ty - dy
            if not (0 <= target_robot_x < self.state.width and 0 <= target_robot_y < self.state.height):
                continue
            if (target_robot_x, target_robot_y) in self.scheduler.pallets and (
                target_robot_x, target_robot_y
            ) != pallet_xy:
                continue

            carry_path = self._safe_plan_path(temp_robot, target_robot_x, target_robot_y)
            if not carry_path and (temp_robot.x != target_robot_x or temp_robot.y != target_robot_y):
                continue

            chosen_target = (tx, ty, carry_path)
            break

        if chosen_target is None:
            return None

        target_pallet_x, target_pallet_y, carry_path = chosen_target
        if carry_path:
            pending_paths.append((self._clone_robot_state(temp_robot), carry_path))
        pending_actions.extend(self._apply_moves_to_robot(temp_robot, carry_path))

        undock_t = temp_robot.last_t + 1
        if not self.planner.can_occupy(temp_robot, undock_t, temp_robot.x, temp_robot.y):
            return None
        pending_actions.append((undock_t, temp_robot.id, "undock", target_pallet_x, target_pallet_y))
        pending_footprints.append((self._clone_robot_state(temp_robot), undock_t, temp_robot.x, temp_robot.y))
        temp_robot.last_t = undock_t
        del temp_robot.docks[(dx, dy)]

        pending_static_additions.append((undock_t + 1, target_pallet_x, target_pallet_y))
        self._commit_plan(
            robot=robot,
            temp_robot=temp_robot,
            pending_actions=pending_actions,
            pending_paths=pending_paths,
            pending_footprints=pending_footprints,
            pending_static_additions=pending_static_additions,
        )
        return (target_pallet_x, target_pallet_y), dock_t, undock_t

    def _finalize_relocation_pallet_state(
        self,
        *,
        pallet_id: int,
        old_xy: Tuple[int, int],
        new_xy: Tuple[int, int],
        dock_t: int,
        undock_t: int,
    ) -> None:
        if new_xy == old_xy:
            return
        self._record_pallet_move(
            pallet_id=pallet_id,
            old_xy=old_xy,
            new_xy=new_xy,
            dock_t=dock_t,
            undock_t=undock_t,
        )
        self.scheduler.move_pallet(old_xy, new_xy)
        self.pallet_id_by_coord.pop(old_xy, None)
        self.pallet_id_by_coord[new_xy] = pallet_id
        self.pallet_by_id[pallet_id]["x"] = new_xy[0]
        self.pallet_by_id[pallet_id]["y"] = new_xy[1]
        self.relocated_pallet_targets.add(new_xy)

    def _plan_relocate_pallet_for_robot(self, robot: RobotState, job: RelocationJob) -> bool:
        source = self._select_relocation_source_pallet(robot, job.sku)
        if source is None:
            return False
        pallet_xy, pallet_id = source

        stand_cells = self._candidate_relocation_stand_cells(robot, pallet_xy)
        if not stand_cells:
            return False
        target_pallet_cells = self._ranked_relocation_target_cells(job, pallet_xy)

        for stand_xy in stand_cells:
            outcome = self._attempt_relocation_via_stand(
                robot=robot,
                pallet_xy=pallet_xy,
                pallet_id=pallet_id,
                stand_xy=stand_xy,
                target_pallet_cells=target_pallet_cells,
            )
            if outcome is None:
                continue
            new_xy, dock_t, undock_t = outcome
            self._finalize_relocation_pallet_state(
                pallet_id=pallet_id,
                old_xy=pallet_xy,
                new_xy=new_xy,
                dock_t=dock_t,
                undock_t=undock_t,
            )
            return True

        return False

    def _knight_cells_around(self, origin: Tuple[int, int]) -> List[Tuple[int, int]]:
        ox, oy = origin
        out = []
        for dx, dy in KNIGHT_OFFSETS:
            tx, ty = ox + dx, oy + dy
            if 0 <= tx < self.state.width and 0 <= ty < self.state.height:
                out.append((tx, ty))
        return out

    def _edge_offset_candidates(
        self, bucket: str, max_depth: int = 6, max_span: int = 10
    ) -> List[Tuple[int, int]]:
        if bucket.startswith("top_"):
            inward = (0, 1)
        elif bucket.startswith("bottom_"):
            inward = (0, -1)
        elif bucket == "left_edge":
            inward = (1, 0)
        elif bucket == "right_edge":
            inward = (-1, 0)
        else:
            return list(KNIGHT_OFFSETS)

        lateral_spans = [0]
        for span in range(1, max_span + 1):
            lateral_spans.extend([span, -span])

        offsets: List[Tuple[int, int]] = []
        for depth in range(1, max_depth + 1):
            for span in lateral_spans:
                if inward[0] == 0:
                    offsets.append((span, depth * inward[1]))
                else:
                    offsets.append((depth * inward[0], span))
        return offsets

    def _choose_unique_relocation_offset(
        self,
        bucket: str,
        hotspot: Tuple[int, int],
        reserved_targets: set[Tuple[int, int]],
    ) -> Tuple[int, int]:
        candidates = self._edge_offset_candidates(bucket, max_depth=8, max_span=12)
        fallback = (0, 0)
        for dx, dy in candidates:
            tx, ty = hotspot[0] + dx, hotspot[1] + dy
            if not (0 <= tx < self.state.width and 0 <= ty < self.state.height):
                continue
            if (tx, ty) in self.scheduler.pallets:
                continue
            if (tx, ty) in self.travel_lane_cells:
                continue
            if (tx, ty) in reserved_targets:
                continue
            return (dx, dy)

        for dx, dy in candidates:
            tx, ty = hotspot[0] + dx, hotspot[1] + dy
            if 0 <= tx < self.state.width and 0 <= ty < self.state.height:
                if (tx, ty) in self.travel_lane_cells:
                    continue
                fallback = (dx, dy)
                break
        return fallback

    def _build_travel_lane_cells(self, lane_width: int) -> set[Tuple[int, int]]:
        lane_width = max(0, int(lane_width))
        cells: set[Tuple[int, int]] = set()

        for hx, hy in FULFILL_HOT_SPOTS:
            if hy in (0, self.state.height - 1):
                xmin = max(0, hx - lane_width)
                xmax = min(self.state.width - 1, hx + lane_width)
                for x in range(xmin, xmax + 1):
                    for y in range(self.state.height):
                        cells.add((x, y))
            if hx in (0, self.state.width - 1):
                ymin = max(0, hy - lane_width)
                ymax = min(self.state.height - 1, hy + lane_width)
                for y in range(ymin, ymax + 1):
                    for x in range(self.state.width):
                        cells.add((x, y))

            for dx in range(-lane_width, lane_width + 1):
                for dy in range(-lane_width, lane_width + 1):
                    tx, ty = hx + dx, hy + dy
                    if 0 <= tx < self.state.width and 0 <= ty < self.state.height:
                        cells.add((tx, ty))

        # Reinforce lanes using metadata high-use cells (with a 1-cell Manhattan halo).
        for x, y in self._metadata_high_use_cells:
            cells.add((x, y))
            for nx, ny in adjacent_cells(self.state.width, self.state.height, x, y):
                cells.add((nx, ny))

        return cells

    def _score_relocation_target(
        self,
        old_xy: Tuple[int, int],
        target_xy: Tuple[int, int],
        hotspot: Tuple[int, int],
    ) -> float:
        if target_xy == old_xy:
            return -10_000.0

        old_dist = abs(old_xy[0] - hotspot[0]) + abs(old_xy[1] - hotspot[1])
        new_dist = abs(target_xy[0] - hotspot[0]) + abs(target_xy[1] - hotspot[1])
        demand_gain = float(old_dist - new_dist)

        lane_penalty = 0.0
        if target_xy in self.travel_lane_cells:
            lane_penalty = 500.0

        occupied_neighbors = 0
        free_neighbors = 0
        for nx, ny in adjacent_cells(self.state.width, self.state.height, target_xy[0], target_xy[1]):
            occupied = (nx, ny) in self.scheduler.pallets and (nx, ny) != old_xy
            if occupied:
                occupied_neighbors += 1
            else:
                free_neighbors += 1

        choke_penalty = float(occupied_neighbors * 30)
        if free_neighbors < 2:
            choke_penalty += 200.0

        density_penalty = 0.0
        for (px, py) in self.scheduler.pallets.keys():
            if (px, py) == old_xy:
                continue
            if abs(px - target_xy[0]) + abs(py - target_xy[1]) <= 2:
                density_penalty += 12.0

        spacing_penalty = 0.0
        if self.relocated_pallet_targets:
            min_dist = min(
                abs(rx - target_xy[0]) + abs(ry - target_xy[1])
                for rx, ry in self.relocated_pallet_targets
            )
            if min_dist < 2:
                spacing_penalty = 180.0

        move_cost_penalty = float(abs(target_xy[0] - old_xy[0]) + abs(target_xy[1] - old_xy[1])) * 0.5
        return (demand_gain * 20.0) - lane_penalty - choke_penalty - density_penalty - spacing_penalty - move_cost_penalty

    def _candidate_relocation_targets(self, job: RelocationJob) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        seen: set[Tuple[int, int]] = set()

        def push(cell: Tuple[int, int]) -> None:
            if cell in seen:
                return
            cx, cy = cell
            if not (0 <= cx < self.state.width and 0 <= cy < self.state.height):
                return
            seen.add(cell)
            out.append(cell)

        if job.preferred_target_xy is not None:
            push(job.preferred_target_xy)

        for cell in self._metadata_candidate_targets_for_sku(job.sku):
            push(cell)

        for dx, dy in self._edge_offset_candidates(job.bucket, max_depth=8, max_span=14):
            push((job.hotspot[0] + dx, job.hotspot[1] + dy))

        for cell in self._knight_cells_around(job.hotspot):
            push(cell)

        return out

    def _record_pallet_move(
        self,
        pallet_id: int,
        old_xy: Tuple[int, int],
        new_xy: Tuple[int, int],
        dock_t: int,
        undock_t: int,
    ) -> None:
        self.pallet_moves[pallet_id].append(
            PalletMove(old_xy=old_xy, new_xy=new_xy, dock_t=dock_t, undock_t=undock_t)
        )
        self.pallet_moves[pallet_id].sort(key=lambda m: m.dock_t)

    def _pallet_static_xy_at(self, pallet_id: int, timestep: int) -> Tuple[int, int] | None:
        current_xy = self.pallet_initial_xy[pallet_id]
        for move in self.pallet_moves.get(pallet_id, []):
            if timestep < move.dock_t:
                return current_xy
            if move.dock_t <= timestep <= move.undock_t:
                return None
            current_xy = move.new_xy
        return current_xy

    def _is_pick_target_static_at_time(self, pallet_xy: Tuple[int, int], pick_t: int) -> bool:
        pallet_id = self.pallet_id_by_coord.get(pallet_xy)
        if pallet_id is None:
            return False
        static_xy = self._pallet_static_xy_at(pallet_id, pick_t)
        return static_xy == pallet_xy

    def _safe_plan_path(
        self, robot: RobotState, target_x: int, target_y: int
    ) -> List[Tuple[int, int, int]]:
        """Adapter boundary for planner pathfinding (Space-Time A* in ReservationPlanner)."""
        return self.planner.plan_path(
            robot,
            target_x,
            target_y,
            max_path_steps=self.config.path_step_limit,
        )

    def _commit_plan(
        self,
        robot: RobotState,
        temp_robot: RobotState,
        pending_actions: List[Tuple[int, int, str, int, int]],
        pending_paths: List[Tuple[RobotState, List[Tuple[int, int, int]]]],
        pending_footprints: List[Tuple[RobotState, int, int, int]],
        pending_static_additions: List[Tuple[int, int, int]] | None = None,
    ) -> None:
        for t, rid, action, x, y in pending_actions:
            self.actions.add(t, rid, action, x, y)

        for path_robot, path in pending_paths:
            self.planner.reserve_path(path_robot, path)

        for foot_robot, t, x, y in pending_footprints:
            self.planner.reserve_footprint(foot_robot, t, x, y)

        if pending_static_additions:
            for t, x, y in pending_static_additions:
                self.planner.add_static_obstacle_from(t, x, y)

        robot.x = temp_robot.x
        robot.y = temp_robot.y
        robot.last_t = temp_robot.last_t
        robot.storage = collections.Counter(temp_robot.storage)
        robot.docks = dict(temp_robot.docks)

    def _apply_moves_to_robot(
        self, robot: RobotState, path: List[Tuple[int, int, int]]
    ) -> List[Tuple[int, int, str, int, int]]:
        emitted: List[Tuple[int, int, str, int, int]] = []
        if not path:
            return emitted

        prev_x, prev_y = robot.x, robot.y
        for t, x, y in path:
            # Space-time A* can emit WAIT steps (same x/y). In the submission format,
            # waiting is represented by omitting an action for that robot/timestep.
            if x != prev_x or y != prev_y:
                emitted.append((t, robot.id, "move", x, y))
            prev_x, prev_y = x, y

        robot.last_t = path[-1][0]
        robot.x = path[-1][1]
        robot.y = path[-1][2]
        return emitted

    def _clone_robot_state(self, robot: RobotState) -> RobotState:
        return RobotState(
            id=robot.id,
            x=robot.x,
            y=robot.y,
            last_t=robot.last_t,
            storage=collections.Counter(robot.storage),
            docks=dict(robot.docks),
        )

    def _lns_improve_actions(
        self, actions: List[Tuple[int, int, str, int, int]]
    ) -> List[Tuple[int, int, str, int, int]]:
        """
        Algorithm: Large Neighborhood Search (LNS).
        Repeatedly samples a tail-window shift neighborhood and accepts improving valid moves.
        """
        if not self.config.lns_enabled:
            return actions
        if self.config.lns_iterations <= 0:
            return actions
        if len(actions) < 4:
            return actions
        if self.config.max_plan_time_seconds > 0 and self._elapsed_plan_seconds() >= self.config.max_plan_time_seconds:
            return actions

        best = list(actions)
        best_makespan = max((t for t, _, _, _, _ in best), default=-1)
        attempts = 0
        accepted = 0
        for _ in range(self.config.lns_iterations):
            if self.config.max_plan_time_seconds > 0 and self._elapsed_plan_seconds() >= self.config.max_plan_time_seconds:
                break
            candidate = self._lns_shift_candidate(best)
            if candidate is None:
                continue
            attempts += 1
            candidate_makespan = max((t for t, _, _, _, _ in candidate), default=-1)
            if candidate_makespan >= best_makespan:
                continue
            if not self._validate_candidate_actions(candidate):
                continue
            best = candidate
            best_makespan = candidate_makespan
            accepted += 1

        if attempts > 0:
            self._log(
                f"lns_done attempts={attempts} accepted={accepted} best_makespan={best_makespan}"
            )
        return best

    def _lns_shift_candidate(
        self, actions: List[Tuple[int, int, str, int, int]]
    ) -> List[Tuple[int, int, str, int, int]] | None:
        by_robot: Dict[int, List[int]] = collections.defaultdict(list)
        for idx, (_, rid, _, _, _) in enumerate(actions):
            by_robot[rid].append(idx)
        if not by_robot:
            return None

        robot_tail_times = [
            (actions[indexes[-1]][0], rid)
            for rid, indexes in by_robot.items()
            if indexes
        ]
        if not robot_tail_times:
            return None
        robot_tail_times.sort(reverse=True)
        top_k = max(1, min(3, len(robot_tail_times)))
        _, robot_id = self.rng.choice(robot_tail_times[:top_k])
        robot_idxs = by_robot[robot_id]
        if len(robot_idxs) < 2:
            return None

        tail_start = int(len(robot_idxs) * (1.0 - self.config.lns_tail_fraction))
        tail_start = max(1, min(len(robot_idxs) - 1, tail_start))
        end_pos = self.rng.randint(tail_start, len(robot_idxs) - 1)
        start_pos = max(0, end_pos - max(1, self.config.lns_window_actions) + 1)
        selected_positions = list(range(start_pos, end_pos + 1))
        if not selected_positions:
            return None

        first_pos = selected_positions[0]
        first_action_index = robot_idxs[first_pos]
        first_t = actions[first_action_index][0]
        prev_t = actions[robot_idxs[first_pos - 1]][0] if first_pos > 0 else -1
        available_gap = first_t - prev_t - 1
        if available_gap <= 0:
            return None
        max_shift = min(max(1, self.config.lns_max_shift), available_gap)
        shift = self.rng.randint(1, max_shift)

        selected_action_indexes = {robot_idxs[pos] for pos in selected_positions}
        shifted: List[Tuple[int, int, str, int, int]] = []
        for idx, row in enumerate(actions):
            t, rid, action, x, y = row
            if idx in selected_action_indexes:
                shifted.append((t - shift, rid, action, x, y))
            else:
                shifted.append(row)

        if not self._has_unique_robot_timestep_pairs(shifted):
            return None
        shifted.sort(key=lambda row: (row[0], row[1]))
        return shifted

    def _has_unique_robot_timestep_pairs(
        self, actions: List[Tuple[int, int, str, int, int]]
    ) -> bool:
        seen: set[Tuple[int, int]] = set()
        for t, rid, _, _, _ in actions:
            key = (t, rid)
            if key in seen:
                return False
            seen.add(key)
        return True

    def _validate_candidate_actions(
        self,
        actions: List[Tuple[int, int, str, int, int]],
        log_on_error: bool = False,
    ) -> bool:
        validator = SubmissionValidator(worklist_text=self._worklist_text_from_state())
        try:
            for t, rid, action, x, y in actions:
                validator.validate_line(f"{t} {rid} {action} {x} {y}")
            final_state = validator.finalize()
        except ValidationError as exc:
            if log_on_error:
                self._log(f"candidate_validation_error: {exc}")
            return False
        ok = final_state.fulfilled_orders == final_state.total_orders
        if log_on_error and not ok:
            self._log(
                f"candidate_validation_incomplete fulfilled={final_state.fulfilled_orders} "
                f"total={final_state.total_orders} next_t={final_state.next_timestep}"
            )
        return ok

    def _worklist_text_from_state(self) -> str:
        lines: List[str] = []
        lines.append(str(len(self.state.robots)))
        for x, y in self.state.robots:
            lines.append(f"{x} {y}")
        lines.append(str(len(self.state.pallets)))
        for (x, y), sku in self.state.pallets.items():
            lines.append(f"{x} {y} {sku}")
        lines.append(str(len(self.state.orders)))
        for order in self.state.orders:
            sku_stream: List[str] = []
            if isinstance(order, collections.Counter):
                items_iter = sorted(order.items())
            else:
                items_iter = sorted(collections.Counter(order).items())
            for sku, qty in items_iter:
                sku_stream.extend([str(sku)] * int(qty))
            if not sku_stream:
                sku_stream.append("0")
            lines.append(" ".join(sku_stream))
        return "\n".join(lines) + "\n"

    def _open_log(self) -> None:
        if self.config.log_path is None:
            return
        log_path = Path(self.config.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("w", encoding="utf-8", buffering=1)

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _log(self, message: str) -> None:
        if self._log_handle is None:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._log_handle.write(f"[{ts}] {message}\n")

    def _repair_idle_wait_conflicts(
        self, actions: List[Tuple[int, int, str, int, int]]
    ) -> List[Tuple[int, int, str, int, int]]:
        """
        Repair a specific invalid pattern:
        A robot moves into a cell occupied by another robot that is implicitly waiting
        there forever (no action at this timestep and no future actions).
        """
        repaired = list(actions)
        static_blocked = (self.state.grid == 2)
        max_repairs = 20

        def footprint_at(x: int, y: int, dock_offsets: set[Tuple[int, int]]) -> set[Tuple[int, int]]:
            fp = {(x, y)}
            for dx, dy in dock_offsets:
                fp.add((x + dx, y + dy))
            return fp

        for _ in range(max_repairs):
            repaired.sort(key=lambda row: (row[0], row[1]))
            by_t = collections.defaultdict(dict)
            robot_times = collections.defaultdict(list)
            for t, rid, action, x, y in repaired:
                by_t[t][rid] = (action, x, y)
                robot_times[rid].append(t)

            positions = {rid: (x, y) for rid, (x, y) in enumerate(self.state.robots)}
            docks: Dict[int, set[Tuple[int, int]]] = {
                rid: set() for rid in range(len(self.state.robots))
            }
            conflict = None

            max_t = max(by_t) if by_t else -1
            for t in range(max_t + 1):
                acts = by_t.get(t, {})
                start_positions = dict(positions)
                start_docks = {rid: set(offsets) for rid, offsets in docks.items()}
                start_footprints = {
                    rid: footprint_at(pos[0], pos[1], start_docks.get(rid, set()))
                    for rid, pos in start_positions.items()
                }
                move_target_footprints = {
                    rid: footprint_at(x, y, start_docks.get(rid, set()))
                    for rid, (action, x, y) in acts.items()
                    if action == "move"
                }
                occupied_start = set()
                for rid, fp in start_footprints.items():
                    occupied_start.update(fp)

                for rid, (action, x, y) in acts.items():
                    if action != "move":
                        continue
                    mover_target = move_target_footprints.get(rid, set())
                    overlapping_blockers = [
                        blocker
                        for blocker, blocker_fp in start_footprints.items()
                        if blocker != rid and mover_target.intersection(blocker_fp)
                    ]
                    if not overlapping_blockers:
                        continue

                    for blocker in overlapping_blockers:
                        blocker_action = acts.get(blocker)
                        blocker_moves = blocker_action is not None and blocker_action[0] == "move"
                        if blocker_moves:
                            continue

                        blocker_has_action_now = blocker_action is not None
                        blocker_has_future = any(tt > t for tt in robot_times.get(blocker, []))
                        if blocker_has_action_now or blocker_has_future:
                            continue

                        bx, by = start_positions[blocker]
                        blocker_docks = start_docks.get(blocker, set())
                        candidates = [(bx - 1, by), (bx + 1, by), (bx, by - 1), (bx, by + 1)]
                        chosen = None
                        for nx, ny in candidates:
                            candidate_fp = footprint_at(nx, ny, blocker_docks)
                            in_bounds = all(
                                0 <= fx < self.state.width and 0 <= fy < self.state.height
                                for fx, fy in candidate_fp
                            )
                            if not in_bounds:
                                continue
                            if any(static_blocked[fy, fx] for fx, fy in candidate_fp):
                                continue

                            occupied_without_blocker = occupied_start - start_footprints[blocker]
                            if candidate_fp.intersection(occupied_without_blocker):
                                continue

                            overlaps_move_target = False
                            for target_rid, target_fp in move_target_footprints.items():
                                if target_rid == blocker:
                                    continue
                                if candidate_fp.intersection(target_fp):
                                    overlaps_move_target = True
                                    break
                            if overlaps_move_target:
                                continue

                            chosen = (nx, ny)
                            break

                        if chosen is None:
                            continue

                        conflict = (t, blocker, chosen[0], chosen[1])
                        break

                    if conflict is not None:
                        break

                for rid, (action, x, y) in acts.items():
                    if action == "move":
                        positions[rid] = (x, y)
                for rid, (action, x, y) in acts.items():
                    if action == "dock":
                        rx, ry = positions[rid]
                        docks[rid].add((x - rx, y - ry))
                    elif action == "undock":
                        rx, ry = positions[rid]
                        docks[rid].discard((x - rx, y - ry))

                if conflict is not None:
                    break

            if conflict is None:
                repaired.sort(key=lambda row: (row[0], row[1]))
                return repaired

            t, blocker, nx, ny = conflict
            repaired.append((t, blocker, "move", nx, ny))

        repaired.sort(key=lambda row: (row[0], row[1]))
        return repaired
