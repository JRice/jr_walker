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

from jr_walker.logic import (
    DockSuggestion,
    EdgeAwareOrderScorer,
    OrderOptimizer,
    OrderSuggestion,
    RelocateSuggestion,
    SetupSuggestion,
    Suggestion,
    manhattan_distance,
)
from jr_walker.hierarchical import MiniBoxMotionPlanner, SetupTaskPlanner
from jr_walker.planner import adjacent_cells
from jr_walker.planner import ReservationPlanner
from jr_walker.scheduler import GreedyScheduler
from jr_walker.sim import ActionLog, RobotState
from jr_walker.validator import SubmissionValidator, ValidationError
from jr_walker.writer import write_actions

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
class SetupJob:
    sku: int
    hotspot: Tuple[int, int]
    source_pallet_id: int
    source_xy: Tuple[int, int]
    target_xy: Tuple[int, int]


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

@dataclass
class PastRunAnalysis:
    run_id: int = -1
    use_by_cell: Dict[Tuple[int, int], int] = field(default_factory=dict)
    sku_cells: Dict[int, List[Tuple[int, int, int]]] = field(default_factory=dict)
    high_use_cells: set[Tuple[int, int]] = field(default_factory=set)
    bucket_items: Dict[str, int] = field(default_factory=dict)
    bucket_sku_counts: Dict[str, collections.Counter] = field(default_factory=dict)
    fulfills: List[Dict] = field(default_factory=list)

@dataclass
class SolverConfig:
    max_time: int = 50000
    max_makespan: int | None = None
    max_plan_time_seconds: float = 600.0
    progress_every: int = 50
    output_path: Path = Path("output/solution.txt")
    random_seed: int = 7
    max_delivery_order_attempts: int = 40
    delivery_candidate_window: int = 160
    path_step_limit: int = 350
    relocate_stand_candidate_limit: int = 6
    relocate_target_candidate_limit: int = 8
    relocation_edge_band: int = 6
    relocate_chunk_size: int = 1
    relocation_analysis_path: Path | None = None
    relocation_top_skus: int = 8
    relocation_min_lift: float = 0.08
    relocation_max_attempts_per_sku: int = 5
    relocation_skus_to_relocate: List[int] | None = None
    num_allowed_relocations: int = 10
    order_suggestion_gain_constant: float = 100.0
    dock_gain_scale: float = 2.0
    relocation_gain_scale: float = 1.5
    strict_no_swap: bool = False
    lane_width: int = 3
    min_jobs_for_dock: int = 3
    log_path: Path | None = None
    dispatch_log_every: int = 1
    dispatch_validate_every_makespan: int = 500
    worklist_path: Path = Path("docs/BIG_ORDER.txt")
    lns_enabled: bool = True
    lns_iterations: int = 60
    lns_window_actions: int = 28
    lns_tail_fraction: float = 0.35
    lns_max_shift: int = 2
    dump_suggestions_path: Path | None = None
    astar_slow_ms: float = 40.0
    astar_print_slow: bool = False
    astar_log_blocked: bool = False
    enable_relocation_suggestions: bool = False
    setup_hotspots: List[Tuple[int, int]] = field(default_factory=list)
    setup_mini_box_radius: int = 2
    suggestion_retry_limit: int = 12
    suggestion_backoff_base_cycles: int = 2
    suggestion_backoff_max_cycles: int = 128
    max_robots_per_suggestion: int = 3
    robot_fail_streak_for_parking: int = 3
    parking_candidate_limit: int = 96


def _select_best_non_test_run_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        """
        SELECT run_id
        FROM metadata_runs
        WHERE solution_path NOT LIKE '%test_solution_%'
          AND solution_path NOT LIKE '%stride_%_solution_%'
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


def _bucket_for_edge(x: int, y: int, width: int, height: int) -> str:
    if x == 0:
        return "left_edge"
    if x == width - 1:
        return "right_edge"
    if y == 0:
        return "top_x0_29" if x < (width // 2) else "top_x30_59"
    if y == height - 1:
        return "bottom_x0_29" if x < (width // 2) else "bottom_x30_59"
    return "non_edge"


def load_best_past_analysis(db_path: Path, width: int, height: int, pallets: Dict[Tuple[int, int], int]) -> PastRunAnalysis:
    analysis = PastRunAnalysis()
    if not db_path.exists():
        return analysis

    try:
        conn = sqlite3.connect(db_path)
        run_id = _select_best_non_test_run_id(conn)
        if run_id is None:
            return analysis
        analysis.run_id = run_id

        use_rows = conn.execute("SELECT x, y, use_score FROM cell_metadata WHERE run_id = ?", (run_id,)).fetchall()
        for x, y, use_score in use_rows:
            analysis.use_by_cell[(int(x), int(y))] = int(use_score)

        sku_rows = conn.execute("SELECT x, y, sku, count FROM cell_sku_flow WHERE run_id = ?", (run_id,)).fetchall()
        for x, y, sku, count in sku_rows:
            analysis.sku_cells.setdefault(int(sku), []).append((int(x), int(y), int(count)))

        for sku, rows in analysis.sku_cells.items():
            rows.sort(key=lambda row: (-row[2], analysis.use_by_cell.get((row[0], row[1]), 0), row[1], row[0]))

        dynamic_use_values: List[int] = []
        for (x, y), use_score in analysis.use_by_cell.items():
            if use_score <= 0 or (x, y) in pallets:
                continue
            dynamic_use_values.append(use_score)

        if dynamic_use_values:
            dynamic_use_values.sort()
            idx = int((len(dynamic_use_values) - 1) * 0.85)
            cutoff = dynamic_use_values[max(0, idx)]
            for (x, y), use_score in analysis.use_by_cell.items():
                if use_score >= cutoff and (x, y) not in pallets:
                    analysis.high_use_cells.add((x, y))

        try:
            fulfill_rows = conn.execute(
                """
                SELECT robot_id, timestep, order_id, x, y, skus_json
                FROM fulfills
                WHERE run_id = ?
                ORDER BY robot_id, timestep ASC
                """,
                (run_id,),
            ).fetchall()
        except sqlite3.Error:
            # Backward-compat for older schemas without order_id.
            legacy_rows = conn.execute(
                """
                SELECT robot_id, timestep, x, y, skus_json
                FROM fulfills
                WHERE run_id = ?
                ORDER BY robot_id, timestep ASC
                """,
                (run_id,),
            ).fetchall()
            fulfill_rows = [(rid, t, None, x, y, skus_json) for rid, t, x, y, skus_json in legacy_rows]

        for _, _, _, x, y, skus_json in fulfill_rows:
            bucket = _bucket_for_edge(int(x), int(y), width, height)
            bucket_counter = analysis.bucket_sku_counts.setdefault(bucket, collections.Counter())
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
                analysis.bucket_items[bucket] = analysis.bucket_items.get(bucket, 0) + 1

        for robot_id, timestep, order_id, _, _, skus_json in fulfill_rows:
            analysis.fulfills.append(
                {
                    "robot_id": robot_id,
                    "timestep": timestep,
                    "order_id": order_id,
                    "skus": json.loads(skus_json),
                }
            )

    except sqlite3.Error:
        pass
    finally:
        if 'conn' in locals():
            conn.close()
    return analysis


class WarehouseSolver:
    def __init__(self, warehouse_state, config: SolverConfig | None = None, past_analysis: PastRunAnalysis | None = None):
        self.state = warehouse_state
        self.config = config or SolverConfig()
        self.rng = random.Random(self.config.random_seed)
        self.past_analysis = past_analysis or PastRunAnalysis()
        self._sku_anchor_rows_cache: Dict[int, List[Tuple[int, int, int]]] = {}

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
        self.travel_lane_cells: set[Tuple[int, int]] = self._build_travel_lane_cells(
            lane_width=self.config.lane_width
        )
        self.task_planner = SetupTaskPlanner()
        self.motion_planner = MiniBoxMotionPlanner(
            width=self.state.width,
            height=self.state.height,
            box_radius=max(1, int(getattr(self.config, "setup_mini_box_radius", 2))),
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
        self.docked_skus: set[int] = set()
        self.relocated_pallet_targets: set[Tuple[int, int]] = set()
        self._completed_order_indices: set[int] = set()
        sku_counter: collections.Counter = collections.Counter()
        for order in self.orders:
            sku_counter.update(order.items)
        self.skus_by_demand: List[int] = [sku for sku, _ in sku_counter.most_common()]
        if self.config.enable_relocation_suggestions:
            self.relocation_plan: Deque[RelocationJob] = self._build_relocation_plan()
        else:
            self.relocation_plan = collections.deque()
        self.setup_jobs: List[SetupJob] = self._build_setup_jobs()
        self._setup_robot_by_hotspot: Dict[Tuple[int, int], int] = self._assign_setup_robots_by_hotspot()
        self._setup_robot_ids: set[int] = set(self._setup_robot_by_hotspot.values())
        self._completed_setup_pallet_ids: set[int] = set()
        self._dropped_setup_pallet_ids: set[int] = set()
        self._setup_job_by_source_pallet_id: Dict[int, SetupJob] = {
            int(job.source_pallet_id): job for job in self.setup_jobs
        }
        self._setup_jobs_by_hotspot: Dict[Tuple[int, int], List[SetupJob]] = collections.defaultdict(list)
        for job in self.setup_jobs:
            hs = (int(job.hotspot[0]), int(job.hotspot[1]))
            self._setup_jobs_by_hotspot[hs].append(job)
        (
            self._setup_slot_index_by_source_pallet_id,
            self._setup_hotspot_frontier_pallet_ids,
        ) = self._build_setup_frontier_maps()
        self._setup_frontier_wait_logged_cycle: Dict[int, int] = {}

        self._plan_started_monotonic = 0.0
        self._log_handle = None
        self._next_dispatch_validation_makespan = max(0, self.config.dispatch_validate_every_makespan)
        self._astar_calls = 0
        self._astar_slow_calls = 0
        self._astar_blocked_calls = 0
        self._astar_total_ms = 0.0
        self._astar_max_ms = 0.0
        self._dispatch_cycle = 0
        self._suggestion_fail_counts: Dict[str, int] = {}
        self._suggestion_backoff_until_cycle: Dict[str, int] = {}
        self._robot_fail_streak: Dict[int, int] = collections.defaultdict(int)
        self._parking_moves = 0

    def _assign_setup_robots_by_hotspot(self) -> Dict[Tuple[int, int], int]:
        hotspots: List[Tuple[int, int]] = []
        seen: set[Tuple[int, int]] = set()
        for job in self.setup_jobs:
            hs = (int(job.hotspot[0]), int(job.hotspot[1]))
            if hs in seen:
                continue
            seen.add(hs)
            hotspots.append(hs)

        if not hotspots:
            return {}

        available = list(self.robots)
        mapping: Dict[Tuple[int, int], int] = {}
        for hx, hy in hotspots:
            if not available:
                break
            best = min(
                available,
                key=lambda r: (
                    abs(r.x - hx) + abs(r.y - hy),
                    r.last_t,
                    r.id,
                ),
            )
            mapping[(hx, hy)] = best.id
            available.remove(best)
        return mapping

    def _build_setup_frontier_maps(
        self,
    ) -> Tuple[Dict[int, int], Dict[Tuple[int, int], List[int]]]:
        slot_index_by_source_pallet_id: Dict[int, int] = {}
        hotspot_frontier_pallet_ids: Dict[Tuple[int, int], List[int]] = {}

        for hotspot, jobs in self._setup_jobs_by_hotspot.items():
            slot_order = {
                cell: idx for idx, cell in enumerate(self._setup_slot_candidates(hotspot, limit=240))
            }
            rows: List[Tuple[int, int]] = []
            for job in jobs:
                source_pallet_id = int(job.source_pallet_id)
                target = (int(job.target_xy[0]), int(job.target_xy[1]))
                slot_idx = int(slot_order.get(target, 10**9))
                slot_index_by_source_pallet_id[source_pallet_id] = slot_idx
                rows.append((slot_idx, source_pallet_id))
            rows.sort(key=lambda row: (row[0], row[1]))
            hotspot_frontier_pallet_ids[hotspot] = [source_pallet_id for _, source_pallet_id in rows]

        return slot_index_by_source_pallet_id, hotspot_frontier_pallet_ids

    def _requeue_setup_suggestion(
        self,
        suggestion_queue: collections.deque[Suggestion],
        suggestion: SetupSuggestion,
    ) -> None:
        for idx, queued in enumerate(suggestion_queue):
            if not isinstance(queued, SetupSuggestion):
                suggestion_queue.insert(idx, suggestion)
                return
        suggestion_queue.append(suggestion)

    def _log_dispatch_stall_state(
        self,
        *,
        current_suggestion: Suggestion,
        suggestion_queue: collections.deque[Suggestion],
        attempts: int,
        queue_size: int,
        completed_orders: int,
        total_orders: int,
        dispatch_count: int,
    ) -> None:
        robot_rows = []
        for robot in self.robots:
            dock_count = len(getattr(robot, "docks", {}) or {})
            storage = getattr(robot, "storage", collections.Counter())
            storage_count = int(sum(storage.values())) if hasattr(storage, "values") else 0
            robot_rows.append(
                f"robot={robot.id} cell=({robot.x},{robot.y}) last_t={robot.last_t} "
                f"docks={dock_count} storage={storage_count}"
            )
        if robot_rows:
            self._log("dispatcher_stall_robots " + " | ".join(robot_rows))

        queue_snapshot: List[Suggestion] = [current_suggestion] + list(suggestion_queue)
        self._log(
            "dispatcher_stall_summary "
            f"attempts={attempts} queue_size={queue_size} "
            f"completed_orders={completed_orders}/{total_orders} dispatches={dispatch_count} "
            f"queue_snapshot_size={len(queue_snapshot)}"
        )
        for idx, queued in enumerate(queue_snapshot):
            self._log(f"dispatcher_stall_queue[{idx}] {queued}")

    def _setup_frontier_blocking_pallet_id(self, job: SetupJob) -> int | None:
        hotspot = (int(job.hotspot[0]), int(job.hotspot[1]))
        current_pid = int(job.source_pallet_id)
        for source_pallet_id in self._setup_hotspot_frontier_pallet_ids.get(hotspot, []):
            if source_pallet_id == current_pid:
                return None
            if source_pallet_id in self._completed_setup_pallet_ids:
                continue
            if source_pallet_id in self._dropped_setup_pallet_ids:
                continue
            return source_pallet_id
        return None

    def _is_setup_frontier_ready(self, job: SetupJob) -> bool:
        return self._setup_frontier_blocking_pallet_id(job) is None

    def _setup_retry_limit(self) -> int:
        return max(1, int(self.config.suggestion_retry_limit))

    def _log_setup_hotspot_progress(self, hotspot: Tuple[int, int], *, reason: str) -> None:
        hotspot_key = (int(hotspot[0]), int(hotspot[1]))
        jobs = self._setup_jobs_by_hotspot.get(hotspot_key, [])
        completed = 0
        dropped = 0
        pending = 0
        next_target: Tuple[int, int] | None = None
        next_source_pallet_id: int | None = None
        for source_pallet_id in self._setup_hotspot_frontier_pallet_ids.get(hotspot_key, []):
            if source_pallet_id in self._completed_setup_pallet_ids:
                completed += 1
                continue
            if source_pallet_id in self._dropped_setup_pallet_ids:
                dropped += 1
                continue
            pending += 1
            if next_target is None:
                job = self._setup_job_by_source_pallet_id.get(source_pallet_id)
                if job is not None:
                    next_target = (int(job.target_xy[0]), int(job.target_xy[1]))
                    next_source_pallet_id = int(job.source_pallet_id)

        # Keep accounting robust if any job is not present in the frontier index map.
        indexed = completed + dropped + pending
        if indexed < len(jobs):
            for job in jobs:
                source_pallet_id = int(job.source_pallet_id)
                if source_pallet_id in self._setup_slot_index_by_source_pallet_id:
                    continue
                if source_pallet_id in self._completed_setup_pallet_ids:
                    completed += 1
                elif source_pallet_id in self._dropped_setup_pallet_ids:
                    dropped += 1
                else:
                    pending += 1
                    if next_target is None:
                        next_target = (int(job.target_xy[0]), int(job.target_xy[1]))
                        next_source_pallet_id = source_pallet_id

        self._log(
            "setup_hotspot_progress "
            f"hotspot={hotspot_key} reason={reason} "
            f"completed={completed} dropped={dropped} pending={pending} "
            f"next_target={next_target} next_source_pallet_id={next_source_pallet_id}"
        )

    @staticmethod
    def _format_reason_counts(counter: collections.Counter, *, limit: int = 5) -> str:
        if not counter:
            return "none"
        rows = counter.most_common(max(1, int(limit)))
        return ",".join(f"{name}:{count}" for name, count in rows)

    def _rebind_setup_source_pallet_id(
        self,
        *,
        job: SetupJob,
        old_source_pallet_id: int,
        new_source_pallet_id: int,
    ) -> None:
        old_source_pallet_id = int(old_source_pallet_id)
        new_source_pallet_id = int(new_source_pallet_id)
        if old_source_pallet_id == new_source_pallet_id:
            return

        hotspot = (int(job.hotspot[0]), int(job.hotspot[1]))
        frontier = self._setup_hotspot_frontier_pallet_ids.get(hotspot, [])
        for idx, source_pallet_id in enumerate(frontier):
            if int(source_pallet_id) == old_source_pallet_id:
                frontier[idx] = new_source_pallet_id
                break

        slot_index = self._setup_slot_index_by_source_pallet_id.pop(old_source_pallet_id, None)
        if slot_index is not None:
            self._setup_slot_index_by_source_pallet_id[new_source_pallet_id] = int(slot_index)
        self._setup_frontier_wait_logged_cycle.pop(old_source_pallet_id, None)

    def _has_pending_setup_for_robot(self, robot_id: int) -> bool:
        setup_jobs = getattr(self, "setup_jobs", [])
        completed = getattr(self, "_completed_setup_pallet_ids", set())
        dropped = getattr(self, "_dropped_setup_pallet_ids", set())
        setup_robot_by_hotspot = getattr(self, "_setup_robot_by_hotspot", {})
        for job in setup_jobs:
            source_pallet_id = getattr(job, "source_pallet_id", None)
            hotspot = getattr(job, "hotspot", None)
            if source_pallet_id is None or hotspot is None:
                continue
            if source_pallet_id in completed:
                continue
            if source_pallet_id in dropped:
                continue
            assigned_id = setup_robot_by_hotspot.get((int(hotspot[0]), int(hotspot[1])))
            if assigned_id == robot_id:
                return True
        return False

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
            return self._optimize_actions_core(list(actions))
        finally:
            self._close_log()

    def _find_solution_actions_core(self) -> List[Tuple[int, int, str, int, int]]:
        """
        Algorithm: greedy suggestion-based dispatcher.
        Pattern: orchestrator loop that iterates through a pre-sorted suggestion queue,
        assigning tasks to the next available robot.
        """
        self._plan_started_monotonic = time.monotonic()

        suggestion_queue = collections.deque(self._build_suggestion_queue())

        total_orders = len(self.orders)
        completed_orders: set[int] = set()
        self._completed_order_indices = completed_orders
        dispatch_count = 0
        no_progress_attempts = 0
        self._log_solve_start(total_orders)

        # Keep trying suggestions until all orders are done or queue is empty.
        while suggestion_queue and len(completed_orders) < total_orders:
            queue_len_snapshot = max(1, len(suggestion_queue))
            self._dispatch_cycle += 1
            self._check_global_limits_or_raise(
                collections.deque(o for o in range(total_orders) if o not in completed_orders)
            )

            def record_no_progress_and_maybe_raise() -> None:
                nonlocal no_progress_attempts
                no_progress_attempts += 1
                if no_progress_attempts < queue_len_snapshot:
                    return
                msg = (
                    "Dispatcher made no progress after scanning the full suggestion queue: "
                    f"attempts={no_progress_attempts} "
                    f"queue_size={queue_len_snapshot} "
                    f"completed_orders={len(completed_orders)}/{total_orders} "
                    f"dispatches={dispatch_count}"
                )
                self._log_dispatch_stall_state(
                    current_suggestion=suggestion,
                    suggestion_queue=suggestion_queue,
                    attempts=no_progress_attempts,
                    queue_size=queue_len_snapshot,
                    completed_orders=len(completed_orders),
                    total_orders=total_orders,
                    dispatch_count=dispatch_count,
                )
                self._log(msg)
                raise RuntimeError(msg)

            suggestion = suggestion_queue.popleft()
            suggestion_key = self._suggestion_key(suggestion)
            blocked_until = self._suggestion_backoff_until_cycle.get(suggestion_key, 0)
            if blocked_until > self._dispatch_cycle:
                if isinstance(suggestion, SetupSuggestion):
                    self._requeue_setup_suggestion(suggestion_queue, suggestion)
                else:
                    suggestion_queue.append(suggestion)
                record_no_progress_and_maybe_raise()
                continue

            # Skip if already handled
            if isinstance(suggestion, OrderSuggestion) and suggestion.order_idx in completed_orders:
                continue
            if isinstance(suggestion, SetupSuggestion) and suggestion.job.source_pallet_id in self._completed_setup_pallet_ids:
                continue
            if isinstance(suggestion, SetupSuggestion) and suggestion.job.source_pallet_id in self._dropped_setup_pallet_ids:
                continue
            if isinstance(suggestion, RelocateSuggestion) and suggestion.job.sku in self.relocated_skus:
                continue
            if isinstance(suggestion, DockSuggestion) and suggestion.sku in self.docked_skus:
                continue

            if isinstance(suggestion, SetupSuggestion) and not self._is_setup_frontier_ready(suggestion.job):
                blocker = self._setup_frontier_blocking_pallet_id(suggestion.job)
                source_pallet_id = int(suggestion.job.source_pallet_id)
                last_logged = self._setup_frontier_wait_logged_cycle.get(source_pallet_id, -10**9)
                if self._dispatch_cycle - last_logged >= 64:
                    self._setup_frontier_wait_logged_cycle[source_pallet_id] = self._dispatch_cycle
                    blocker_job = self._setup_job_by_source_pallet_id.get(int(blocker)) if blocker is not None else None
                    self._log(
                        "setup_frontier_wait "
                        f"hotspot={suggestion.job.hotspot} source_pallet_id={source_pallet_id} "
                        f"target={suggestion.job.target_xy} waiting_on={blocker} "
                        f"waiting_target={(blocker_job.target_xy if blocker_job is not None else None)}"
                    )
                self._requeue_setup_suggestion(suggestion_queue, suggestion)
                record_no_progress_and_maybe_raise()
                continue

            handled = False
            skip_without_requeue = False
            candidate_robots = self._candidate_robots_for_suggestion(suggestion)
            for robot in candidate_robots:
                if isinstance(suggestion, OrderSuggestion):
                    handled = self._plan_order_for_robot(suggestion.order_idx, suggestion.order, robot)
                    if handled:
                        completed_orders.add(suggestion.order_idx)
                elif isinstance(suggestion, SetupSuggestion):
                    handled = self._plan_setup_pallet_for_robot(robot, suggestion.job)
                    if handled:
                        self._completed_setup_pallet_ids.add(suggestion.job.source_pallet_id)
                        self._log(
                            f"setup_done sku={suggestion.job.sku} source={suggestion.job.source_xy} "
                            f"target={suggestion.job.target_xy} hotspot={suggestion.job.hotspot}"
                        )
                        self._log_setup_hotspot_progress(
                            suggestion.job.hotspot,
                            reason=f"done_pid_{suggestion.job.source_pallet_id}",
                        )
                        # Re-score pending order suggestions immediately after each
                        # setup placement so robots can exploit newly moved pallets.
                        suggestion_queue = self._refresh_pending_order_suggestions(
                            suggestion_queue,
                            completed_orders,
                        )
                elif isinstance(suggestion, RelocateSuggestion):
                    if suggestion.remaining_job_factor() <= 0:
                        handled = True
                        skip_without_requeue = True
                        break
                    handled = self._plan_relocate_pallet_for_robot(robot, suggestion.job)
                    if handled:
                        self.relocated_skus.add(suggestion.job.sku)
                        self._log(
                            f"relocation_done sku={suggestion.job.sku} bucket={suggestion.job.bucket} "
                            f"target={suggestion.job.preferred_target_xy} score={suggestion.job.score:.3f}"
                        )
                        # Re-score pending delivery/order suggestions against the new pallet layout.
                        # Keep existing non-order suggestions (relocate/dock) as-is.
                        suggestion_queue = self._refresh_pending_order_suggestions(
                            suggestion_queue,
                            completed_orders,
                        )
                elif isinstance(suggestion, DockSuggestion):
                    handled = self._plan_dock_pallet(robot, suggestion.sku)
                    if handled:
                        self.docked_skus.add(suggestion.sku)
                        self._log(
                            f"dock_done sku={suggestion.sku} center={suggestion.center} "
                            f"gain={suggestion.expected_gain:.3f} score={suggestion.score():.3f}"
                        )
                else:
                    handled = True

                if handled:
                    self._robot_fail_streak[robot.id] = 0
                    break
                self._robot_fail_streak[robot.id] = self._robot_fail_streak.get(robot.id, 0) + 1

            if handled:
                self._suggestion_fail_counts.pop(suggestion_key, None)
                self._suggestion_backoff_until_cycle.pop(suggestion_key, None)
                no_progress_attempts = 0
                if skip_without_requeue:
                    continue
            else:
                parked = False
                if not isinstance(suggestion, SetupSuggestion):
                    for robot in candidate_robots:
                        if self._robot_fail_streak.get(robot.id, 0) < self.config.robot_fail_streak_for_parking:
                            continue
                        if self._plan_idle_parking_move(robot):
                            self._parking_moves += 1
                            parked = True
                            self._robot_fail_streak[robot.id] = 0
                            break

                failure_count = self._suggestion_fail_counts.get(suggestion_key, 0) + 1
                self._suggestion_fail_counts[suggestion_key] = failure_count
                retry_limit = int(self.config.suggestion_retry_limit)
                if isinstance(suggestion, SetupSuggestion):
                    retry_limit = self._setup_retry_limit()
                if failure_count >= retry_limit:
                    self._log(
                        "suggestion_dropped "
                        f"key={suggestion_key} failures={failure_count} parked={parked}"
                    )
                    if isinstance(suggestion, SetupSuggestion):
                        self._dropped_setup_pallet_ids.add(suggestion.job.source_pallet_id)
                        assigned = self._setup_robot_by_hotspot.get(
                            (int(suggestion.job.hotspot[0]), int(suggestion.job.hotspot[1]))
                        )
                        self._log(
                            "setup_job_dropped "
                            f"sku={suggestion.job.sku} hotspot={suggestion.job.hotspot} "
                            f"source_pallet_id={suggestion.job.source_pallet_id} assigned_robot={assigned}"
                        )
                        self._log_setup_hotspot_progress(
                            suggestion.job.hotspot,
                            reason=f"dropped_pid_{suggestion.job.source_pallet_id}",
                        )
                    self._suggestion_backoff_until_cycle.pop(suggestion_key, None)
                else:
                    backoff_cycles = self._suggestion_backoff_cycles(failure_count)
                    self._suggestion_backoff_until_cycle[suggestion_key] = self._dispatch_cycle + backoff_cycles
                    if isinstance(suggestion, SetupSuggestion):
                        self._requeue_setup_suggestion(suggestion_queue, suggestion)
                    else:
                        suggestion_queue.append(suggestion)
                    if failure_count == 1 or failure_count % 3 == 0:
                        self._log(
                            "suggestion_retry_scheduled "
                            f"key={suggestion_key} failures={failure_count} "
                            f"backoff_cycles={backoff_cycles} parked={parked}"
                        )
                record_no_progress_and_maybe_raise()

            if handled:
                no_progress_attempts = 0
                dispatch_count += 1
                self._maybe_log_progress(
                    completed=len(completed_orders),
                    total_orders=total_orders,
                    dispatch_count=dispatch_count,
                )
                self._maybe_validate_dispatch_progress_or_raise(
                    completed=len(completed_orders),
                    total_orders=total_orders,
                    dispatch_count=dispatch_count,
                )

            if dispatch_count > (total_orders + self.config.num_allowed_relocations) * 5:  # circuit breaker
                self._log("Dispatcher seems to be stuck in a loop. Breaking.")
                break

        if len(completed_orders) < total_orders:
            self._log(f"Warning: Only {len(completed_orders)}/{total_orders} orders were completed from suggestion queue.")
        if self.setup_jobs:
            pending = len(self.setup_jobs) - len(self._completed_setup_pallet_ids) - len(self._dropped_setup_pallet_ids)
            self._log(
                "setup_summary "
                f"planned={len(self.setup_jobs)} completed={len(self._completed_setup_pallet_ids)} "
                f"dropped={len(self._dropped_setup_pallet_ids)} pending={max(0, pending)}"
            )

        sorted_actions = self.actions.sorted_actions()
        repaired_actions = self._repair_idle_wait_conflicts(sorted_actions)
        if repaired_actions != sorted_actions and not self._validate_candidate_actions(
            repaired_actions, log_on_error=True
        ):
            self._log("repair_idle_wait_conflicts produced invalid actions; reverting to unrepaired actions")
        else:
            sorted_actions = repaired_actions
        makespan = max((t for t, _, _, _, _ in sorted_actions), default=-1)
        self._log(
            f"find_solution_end actions={len(sorted_actions)} makespan={makespan}"
        )
        if self._astar_calls > 0:
            avg_ms = self._astar_total_ms / self._astar_calls
            self._log(
                "astar_summary "
                f"calls={self._astar_calls} slow_calls={self._astar_slow_calls} "
                f"blocked_calls={self._astar_blocked_calls} avg_ms={avg_ms:.3f} "
                f"max_ms={self._astar_max_ms:.3f}"
            )
        if self._parking_moves > 0:
            self._log(f"idle_parking_summary moves={self._parking_moves}")
        return sorted_actions

    def _build_suggestion_queue(self) -> List[Suggestion]:
        """
        Generates and returns a list of all suggestions.
        SetupSuggestion items are forced to the front while any remain.
        """
        all_suggestions: List[Suggestion] = []
        setup_suggestions: List[Suggestion] = self._build_setup_suggestions()
        all_suggestions.extend(setup_suggestions)
        non_setup_suggestions: List[Suggestion] = []

        # 1. Generate RelocateSuggestions (optional; disabled by default).
        if self.config.enable_relocation_suggestions:
            relocate_suggestions: List[RelocateSuggestion] = []
            relocation_jobs = list(self.relocation_plan)
            for job in relocation_jobs:
                relocate_suggestions.append(
                    RelocateSuggestion(
                        job,
                        self.scheduler,
                        remaining_job_factor_fn=self._count_remaining_orders_for_sku,
                    )
                )

            # Normalize RelocateSuggestion gains to be on a similar scale to OrderSuggestion gains.
            if relocate_suggestions:
                max_relocate_gain = max(s.expected_gain for s in relocate_suggestions)
                if max_relocate_gain > 0:
                    # Scale so the best relocation has a gain comparable to the order constant
                    scaling_factor = (
                        self.config.order_suggestion_gain_constant / max_relocate_gain
                    ) * self.config.relocation_gain_scale
                    self._log(f"Normalizing relocation gains by factor {scaling_factor:.3f}")
                    for s in relocate_suggestions:
                        s.scale_gain(scaling_factor)

            # Sort and truncate RelocateSuggestions
            relocate_suggestions.sort(key=lambda s: s.score(), reverse=True)
            if self.config.num_allowed_relocations >= 0:
                non_setup_suggestions.extend(relocate_suggestions[: self.config.num_allowed_relocations])
        else:
            self._log("RelocateSuggestion generation disabled by config.")

        # 2. Generate DockSuggestions from past run analysis
        if self.past_analysis.fulfills:
            dock_suggestions = self._build_dock_suggestions()
            self._log(f"dock_suggestions count={len(dock_suggestions)} scale={self.config.dock_gain_scale:.3f}")
            non_setup_suggestions.extend(dock_suggestions)

        # 3. Generate OrderSuggestions
        all_order_indexes = list(range(len(self.orders)))
        non_setup_suggestions.extend(self._build_order_suggestions_for_indexes(all_order_indexes))

        # 4. Sort non-setup suggestions only; setup suggestions remain at the front.
        non_setup_suggestions.sort(key=self._suggestion_sort_key)
        all_suggestions.extend(non_setup_suggestions)

        if self.config.dump_suggestions_path:
            dump_data = []
            for s in all_suggestions:
                item = {
                    "score": s.score(),
                    "cost": s.expected_cost,
                    "gain": s.expected_gain,
                    "center": s.center,
                }
                if isinstance(s, OrderSuggestion):
                    item["type"] = "Order"
                    item["order_idx"] = s.order_idx
                elif isinstance(s, RelocateSuggestion):
                    item["type"] = "Relocate"
                    item["sku"] = s.job.sku
                    item["target"] = s.job.preferred_target_xy
                    item["hotspot"] = s.job.hotspot
                    item["bucket"] = s.job.bucket
                dump_data.append(item)

            try:
                import tomli_w
                with open(self.config.dump_suggestions_path, "wb") as f:
                    tomli_w.dump({"suggestion": dump_data}, f)
                self._log(f"dumped {len(dump_data)} suggestions to {self.config.dump_suggestions_path}")
            except ImportError:
                self._log("tomli_w not installed, cannot dump suggestions.")

        return all_suggestions

    def _suggestion_sort_key(self, suggestion: Suggestion) -> Tuple[int, float]:
        # Setup suggestions are always ranked ahead of other suggestion types.
        priority = 0 if isinstance(suggestion, SetupSuggestion) else 1
        return (priority, -suggestion.score())

    def _normalize_setup_hotspots(self) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        seen: set[Tuple[int, int]] = set()
        for raw in self.config.setup_hotspots:
            if not isinstance(raw, (tuple, list)) or len(raw) != 2:
                continue
            try:
                x = int(raw[0])
                y = int(raw[1])
            except (TypeError, ValueError):
                continue
            if not (0 <= x < self.state.width and 0 <= y < self.state.height):
                continue
            # Setup placement strategy is edge-anchored. Non-edge hotspots are
            # projected to their nearest perimeter anchor.
            cell = self._nearest_edge_anchor((x, y))
            if cell in seen:
                continue
            seen.add(cell)
            out.append(cell)
        return out

    def _nearest_edge_anchor(self, cell: Tuple[int, int]) -> Tuple[int, int]:
        x, y = int(cell[0]), int(cell[1])
        candidates = [
            (x, 0),  # top
            (x, self.state.height - 1),  # bottom
            (0, y),  # left
            (self.state.width - 1, y),  # right
        ]
        return min(
            candidates,
            key=lambda p: (
                abs(p[0] - x) + abs(p[1] - y),
                p[1],
                p[0],
            ),
        )

    def _setup_slot_candidates(self, hotspot: Tuple[int, int], limit: int = 240) -> List[Tuple[int, int]]:
        sx, sy = self._nearest_edge_anchor(hotspot)
        out: List[Tuple[int, int]] = []

        # Edge-hotspot template:
        # 1) Along-edge line starting at the hotspot cell.
        # 2) Second line two cells inward from the same hotspot.
        if sy == 0:
            for x in range(sx, self.state.width):
                out.append((x, sy))
                if len(out) >= limit:
                    return out
            inward_y = sy + 2
            if inward_y < self.state.height:
                for x in range(sx, self.state.width):
                    out.append((x, inward_y))
                    if len(out) >= limit:
                        return out
            return out

        if sy == self.state.height - 1:
            for x in range(sx, self.state.width):
                out.append((x, sy))
                if len(out) >= limit:
                    return out
            inward_y = sy - 2
            if inward_y >= 0:
                for x in range(sx, self.state.width):
                    out.append((x, inward_y))
                    if len(out) >= limit:
                        return out
            return out

        if sx == 0:
            for y in range(sy, self.state.height):
                out.append((sx, y))
                if len(out) >= limit:
                    return out
            inward_x = sx + 2
            if inward_x < self.state.width:
                for y in range(sy, self.state.height):
                    out.append((inward_x, y))
                    if len(out) >= limit:
                        return out
            return out

        if sx == self.state.width - 1:
            for y in range(sy, self.state.height):
                out.append((sx, y))
                if len(out) >= limit:
                    return out
            inward_x = sx - 2
            if inward_x >= 0:
                for y in range(sy, self.state.height):
                    out.append((inward_x, y))
                    if len(out) >= limit:
                        return out
            return out
        return out

    def _setup_target_for_hotspot_sku(
        self,
        hotspot: Tuple[int, int],
        sku: int,
        odd_index: Dict[int, int],
        even_index: Dict[int, int],
    ) -> Tuple[int, int] | None:
        hx, hy = self._nearest_edge_anchor(hotspot)

        if sku % 2 == 1:
            idx = odd_index.get(sku)
            if idx is None:
                return None
            if hy == 0 or hy == self.state.height - 1:
                tx, ty = hx + idx, hy
            else:
                tx, ty = hx, hy + idx
        else:
            idx = even_index.get(sku)
            if idx is None:
                return None
            if hy == 0:
                tx, ty = hx + idx, hy + 2
            elif hy == self.state.height - 1:
                tx, ty = hx + idx, hy - 2
            elif hx == 0:
                tx, ty = hx + 2, hy + idx
            else:
                tx, ty = hx - 2, hy + idx

        if not (0 <= tx < self.state.width and 0 <= ty < self.state.height):
            return None
        return (tx, ty)

    def _setup_inward_step(self, hotspot: Tuple[int, int]) -> int:
        _, sy = hotspot
        top_dist = sy
        bottom_dist = self.state.height - 1 - sy
        return -1 if top_dist <= bottom_dist else 1

    def _setup_pull_directions(
        self,
        from_xy: Tuple[int, int],
        target_xy: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        fx, fy = from_xy
        tx, ty = target_xy
        dx = tx - fx
        dy = ty - fy
        dirs: List[Tuple[int, int]] = []

        if abs(dx) >= abs(dy):
            if dx != 0:
                dirs.append((1 if dx > 0 else -1, 0))
            if dy != 0:
                dirs.append((0, 1 if dy > 0 else -1))
        else:
            if dy != 0:
                dirs.append((0, 1 if dy > 0 else -1))
            if dx != 0:
                dirs.append((1 if dx > 0 else -1, 0))
        return dirs

    def _execute_local_pivot_maneuver(
        self,
        *,
        staged_robot: RobotState,
        pallet_id: int,
        staged_pallet_xy: Tuple[int, int],
        staged_offset: Tuple[int, int],
        target_offset: Tuple[int, int],
        staged_paths: List[Tuple[RobotState, List[Tuple[int, int, int]]]],
        staged_actions: List[Tuple[int, int, str, int, int]],
        staged_footprints: List[Tuple[RobotState, int, int, int]],
        note,
    ) -> Tuple[bool, Tuple[int, int]]:
        motion_planner = getattr(self, "motion_planner", None)
        if motion_planner is None:
            motion_planner = MiniBoxMotionPlanner(
                width=self.state.width,
                height=self.state.height,
                box_radius=max(1, int(getattr(self.config, "setup_mini_box_radius", 2))),
            )

        maneuver = motion_planner.plan_pivot(
            robot_xy=(staged_robot.x, staged_robot.y),
            pallet_xy=staged_pallet_xy,
            start_offset=staged_offset,
            target_offset=target_offset,
            static_blocked_cells=self.scheduler.pallets.keys(),
            box_radius=max(1, int(getattr(self.config, "setup_mini_box_radius", 2))),
        )
        if maneuver is None:
            note("redock_maneuver_no_local_plan")
            return False, staged_offset

        current_offset = staged_offset
        for step in maneuver.steps:
            if step.action == "undock":
                undock_t = staged_robot.last_t + 1
                if not self.planner.can_occupy(staged_robot, undock_t, staged_robot.x, staged_robot.y):
                    note("redock_undock_blocked")
                    return False, current_offset
                staged_actions.append((undock_t, staged_robot.id, "undock", step.x, step.y))
                staged_footprints.append(
                    (self._clone_robot_state(staged_robot), undock_t, staged_robot.x, staged_robot.y)
                )
                staged_robot.last_t = undock_t
                staged_robot.docks.pop(current_offset, None)
                continue

            if step.action == "move":
                move_path = self._safe_plan_path_with_step_cap(
                    staged_robot,
                    step.x,
                    step.y,
                    max_path_steps=1,
                )
                if not move_path and (staged_robot.x != step.x or staged_robot.y != step.y):
                    note("redock_local_step_blocked")
                    return False, current_offset
                if move_path:
                    staged_paths.append((self._clone_robot_state(staged_robot), move_path))
                staged_actions.extend(self._apply_moves_to_robot(staged_robot, move_path))
                continue

            if step.action == "dock":
                dock_dx = step.x - staged_robot.x
                dock_dy = step.y - staged_robot.y
                if abs(dock_dx) + abs(dock_dy) != 1:
                    note("redock_not_adjacent")
                    return False, current_offset
                dock_t = staged_robot.last_t + 1
                if not self.planner.can_occupy(staged_robot, dock_t, staged_robot.x, staged_robot.y):
                    note("redock_footprint_blocked")
                    return False, current_offset
                staged_actions.append((dock_t, staged_robot.id, "dock", step.x, step.y))
                staged_footprints.append(
                    (self._clone_robot_state(staged_robot), dock_t, staged_robot.x, staged_robot.y)
                )
                staged_robot.last_t = dock_t
                staged_robot.docks[(dock_dx, dock_dy)] = pallet_id
                current_offset = (dock_dx, dock_dy)
                continue

            note("redock_unknown_step")
            return False, current_offset

        if current_offset != target_offset:
            note("redock_offset_mismatch")
            return False, current_offset
        return True, current_offset

    def _first_available_setup_target(
        self,
        hotspot: Tuple[int, int],
        *,
        reserved_targets: set[Tuple[int, int]],
        source_xy: Tuple[int, int] | None = None,
    ) -> Tuple[int, int] | None:
        for cell in self._setup_slot_candidates(hotspot):
            if cell in reserved_targets:
                continue
            if cell in self.scheduler.pallets and cell != source_xy:
                continue
            return cell
        return None

    def _nearest_unreserved_pallet_for_sku(
        self,
        sku: int,
        hotspot: Tuple[int, int],
        reserved_pallet_ids: set[int],
    ) -> Tuple[Tuple[int, int], int] | None:
        hx, hy = hotspot
        candidates: List[Tuple[int, int, int]] = []
        for cell in self.scheduler.pallet_cells_for_sku(sku):
            pallet_id = self.pallet_id_by_coord.get(cell)
            if pallet_id is None or pallet_id in reserved_pallet_ids:
                continue
            dist = abs(cell[0] - hx) + abs(cell[1] - hy)
            candidates.append((dist, cell[1], cell[0]))
        if not candidates:
            return None
        _, y, x = min(candidates)
        cell = (x, y)
        pallet_id = self.pallet_id_by_coord.get(cell)
        if pallet_id is None:
            return None
        return cell, pallet_id

    def _build_setup_jobs(self) -> List[SetupJob]:
        hotspots = self._normalize_setup_hotspots()
        if not hotspots:
            return []

        all_skus = sorted({int(sku) for sku in self.scheduler.pallets.values()})
        odd_skus = [sku for sku in all_skus if sku % 2 == 1]
        even_skus = [sku for sku in all_skus if sku % 2 == 0]
        odd_index = {sku: idx for idx, sku in enumerate(odd_skus)}
        even_index = {sku: idx for idx, sku in enumerate(even_skus)}
        reserved_pallet_ids: set[int] = set()
        reserved_targets: set[Tuple[int, int]] = set()
        reserved_sources_by_hotspot_sku: Dict[Tuple[Tuple[int, int], int], Tuple[Tuple[int, int], int]] = {}
        jobs: List[SetupJob] = []

        # First pass: reserve the nearest unreserved pallet for each hotspot/SKU.
        for hotspot in hotspots:
            for sku in all_skus:
                source = self._nearest_unreserved_pallet_for_sku(
                    sku,
                    hotspot,
                    reserved_pallet_ids=reserved_pallet_ids,
                )
                if source is None:
                    continue
                source_xy, source_pallet_id = source
                reserved_pallet_ids.add(source_pallet_id)
                reserved_sources_by_hotspot_sku[(hotspot, sku)] = (source_xy, source_pallet_id)

        # Second pass: build setup jobs from reserved hotspot/SKU sources.
        for hotspot in hotspots:
            ordered_skus = odd_skus + even_skus
            for sku in ordered_skus:
                source = reserved_sources_by_hotspot_sku.get((hotspot, sku))
                if source is None:
                    continue
                source_xy, source_pallet_id = source

                target_xy = self._setup_target_for_hotspot_sku(
                    hotspot,
                    sku,
                    odd_index=odd_index,
                    even_index=even_index,
                )
                if target_xy is None:
                    self._log(
                        f"setup_target_unavailable hotspot={hotspot} sku={sku} source={source_xy}"
                    )
                    continue
                if target_xy in reserved_targets and target_xy != source_xy:
                    self._log(
                        f"setup_target_reserved hotspot={hotspot} sku={sku} source={source_xy} target={target_xy}"
                    )
                    continue
                if target_xy in self.scheduler.pallets and target_xy != source_xy:
                    self._log(
                        f"setup_target_occupied hotspot={hotspot} sku={sku} source={source_xy} target={target_xy}"
                    )
                    continue

                reserved_targets.add(target_xy)
                jobs.append(
                    SetupJob(
                        sku=sku,
                        hotspot=hotspot,
                        source_pallet_id=source_pallet_id,
                        source_xy=source_xy,
                        target_xy=target_xy,
                    )
                )

        return jobs

    def _build_setup_suggestions(self) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        for job in self.setup_jobs:
            suggestion = SetupSuggestion(job)
            hotspot_key = (int(job.hotspot[0]), int(job.hotspot[1]))
            assigned_robot_id = self._setup_robot_by_hotspot.get(hotspot_key)
            setattr(suggestion, "assigned_robot_id", assigned_robot_id)
            suggestions.append(suggestion)
        if suggestions:
            self._log(f"setup_suggestions count={len(suggestions)}")
        return suggestions

    def _build_order_suggestions_for_indexes(self, order_indexes: List[int]) -> List[OrderSuggestion]:
        suggestions: List[OrderSuggestion] = []
        if not order_indexes:
            return suggestions

        # Use current scheduler pallet positions (includes relocations).
        optimizer = OrderOptimizer(self.scheduler.pallets)
        scored_orders: List[Tuple[int, int, dict]] = []

        for order_idx in order_indexes:
            order = self.orders[order_idx]
            unique_skus = list(order.items.keys())
            cluster, bbox_score = optimizer.find_tightest_cluster(unique_skus)
            if cluster is None:
                self._log(f"Could not find a pallet cluster for order {order_idx}, skipping suggestion.")
                continue
            scored_orders.append((int(bbox_score), int(order_idx), cluster))

        scored_orders.sort(key=lambda row: (row[0], row[1]))
        for _, order_idx, cluster in scored_orders:
            order = self.orders[order_idx]
            suggestions.append(
                OrderSuggestion(
                    order_idx=order_idx,
                    order=order.items,
                    cluster=cluster,
                    order_gain_constant=self.config.order_suggestion_gain_constant,
                    warehouse_width=self.state.width,
                    warehouse_height=self.state.height,
                    scheduler=self.scheduler,
                )
            )
        return suggestions

    def _refresh_pending_order_suggestions(
        self,
        suggestion_queue: collections.deque[Suggestion],
        completed_orders: set[int],
    ) -> collections.deque[Suggestion]:
        pending_non_order: List[Suggestion] = [
            s for s in suggestion_queue if not isinstance(s, OrderSuggestion)
        ]
        remaining_order_indexes = [
            order.order_idx for order in self.orders if order.order_idx not in completed_orders
        ]
        refreshed_order_suggestions = self._build_order_suggestions_for_indexes(remaining_order_indexes)
        merged: List[Suggestion] = pending_non_order + refreshed_order_suggestions
        merged.sort(key=self._suggestion_sort_key)
        self._log(
            "order_suggestions_refreshed "
            f"remaining_orders={len(remaining_order_indexes)} "
            f"order_suggestions={len(refreshed_order_suggestions)} "
            f"non_order_suggestions={len(pending_non_order)}"
        )
        return collections.deque(merged)

    def _count_remaining_orders_for_sku(self, sku: int) -> int:
        remaining = 0
        for order in self.orders:
            if order.order_idx in self._completed_order_indices:
                continue
            if sku in order.items:
                remaining += 1
        return remaining

    def _build_dock_suggestions(self) -> List[Suggestion]:
        """
        Analyzes past fulfillment data to find "runs" of orders for a single robot
        that share a common, high-demand SKU.
        """
        suggestions: List[Suggestion] = []
        fulfills_by_robot: Dict[int, List[Dict]] = collections.defaultdict(list)
        for f in self.past_analysis.fulfills:
            fulfills_by_robot[f["robot_id"]].append(f)

        optimizer = OrderOptimizer(self.state.pallets)

        for robot_id, fulfills in fulfills_by_robot.items():
            fulfills_sorted = sorted(
                fulfills,
                key=lambda row: int(row.get("timestep", 0)),
            )

            # For each high-demand SKU, look for streaks
            for sku in self.skus_by_demand[:10]:
                current_streak: List[int] = []

                def emit_streak() -> None:
                    if len(current_streak) < max(2, int(self.config.min_jobs_for_dock)):
                        return

                    first_order_idx = current_streak[0]
                    if first_order_idx < 0 or first_order_idx >= len(self.orders):
                        return
                    order_items = self.orders[first_order_idx].items
                    cluster, _ = optimizer.find_tightest_cluster(list(order_items.keys()))
                    if not cluster:
                        return

                    xs = [pos[0] for pos in cluster.values()]
                    ys = [pos[1] for pos in cluster.values()]
                    order_center = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

                    pallet_locs = self.scheduler.pallet_cells_for_sku(sku)
                    if not pallet_locs:
                        return

                    nearest_pallet = min(pallet_locs, key=lambda p: manhattan_distance(p, order_center))
                    gain = (
                        manhattan_distance(order_center, nearest_pallet)
                        * len(current_streak)
                        * self.config.dock_gain_scale
                    )
                    suggestions.append(DockSuggestion(sku, list(current_streak), gain, nearest_pallet))

                for fulfill in fulfills_sorted:
                    order_id = fulfill.get("order_id")
                    if order_id is None or order_id < 0:
                        continue

                    if sku in fulfill["skus"]:
                        current_streak.append(order_id)
                    else:
                        emit_streak()
                        current_streak = []
                # Flush a trailing streak that reaches the end of fulfill history.
                emit_streak()
        return suggestions

    def _optimize_actions_core(
        self, actions: List[Tuple[int, int, str, int, int]]
    ) -> List[Tuple[int, int, str, int, int]]:
        self._plan_started_monotonic = time.monotonic()
        repaired = self._repair_idle_wait_conflicts(list(actions))
        improved = self._lns_improve_actions(repaired)
        makespan = max((t for t, _, _, _, _ in improved), default=-1)
        self._log(f"optimize_end actions={len(improved)} makespan={makespan}")
        return improved


    def _log_solve_start(self, total_orders: int) -> None:
        self._log(f"solve_start total_orders={total_orders}")
        self._log(
            f"setup_plan count={len(self.setup_jobs)} hotspots={self._normalize_setup_hotspots()} "
            f"coverage=nearest_reserved_per_hotspot_sku mini_box_radius={getattr(self.config, 'setup_mini_box_radius', 2)}"
        )
        if self._setup_robot_by_hotspot:
            mapping_summary = ", ".join(
                f"{hotspot}->R{rid}"
                for hotspot, rid in sorted(
                    self._setup_robot_by_hotspot.items(),
                    key=lambda item: (item[0][1], item[0][0], item[1]),
                )
            )
            self._log(f"setup_robot_assignment [{mapping_summary}]")
        if self.setup_jobs:
            jobs_by_hotspot: Dict[Tuple[int, int], List[SetupJob]] = collections.defaultdict(list)
            for job in self.setup_jobs:
                jobs_by_hotspot[(int(job.hotspot[0]), int(job.hotspot[1]))].append(job)

            for hotspot, jobs in sorted(jobs_by_hotspot.items(), key=lambda item: (item[0][1], item[0][0])):
                slot_order = {cell: idx for idx, cell in enumerate(self._setup_slot_candidates(hotspot, limit=240))}
                jobs_sorted = sorted(
                    jobs,
                    key=lambda j: (slot_order.get((int(j.target_xy[0]), int(j.target_xy[1])), 9999), j.sku),
                )
                preview = ", ".join(
                    f"{j.target_xy}:sku{j.sku}:src{j.source_xy}:pid{j.source_pallet_id}"
                    for j in jobs_sorted[:10]
                )
                assigned = self._setup_robot_by_hotspot.get(hotspot)
                self._log(
                    f"setup_hotspot_plan hotspot={hotspot} assigned_robot={assigned} "
                    f"jobs={len(jobs_sorted)} preview=[{preview}]"
                )
                self._log_setup_hotspot_progress(hotspot, reason="initial")
        self._log(
            "dispatch_policy "
            f"retry_limit={self.config.suggestion_retry_limit} "
            f"backoff_base={self.config.suggestion_backoff_base_cycles} "
            f"backoff_max={self.config.suggestion_backoff_max_cycles} "
            f"robots_per_suggestion={self.config.max_robots_per_suggestion} "
            f"parking_fail_streak={self.config.robot_fail_streak_for_parking}"
        )
        self._log(f"relocate_suggestions_enabled={self.config.enable_relocation_suggestions}")
        if self.relocation_plan:
            summary = ", ".join(
                f"SKU{job.sku}->{job.bucket}@{job.placement_offset}(score={job.score:.2f})"
                for job in list(self.relocation_plan)[:6]
            )
            self._log(f"relocation_plan count={len(self.relocation_plan)} top=[{summary}]")
        else:
            self._log("relocation_plan count=0")

    def _maybe_log_progress(self, *, completed: int, total_orders: int, dispatch_count: int) -> None:
        if completed % self.config.progress_every != 0:
            return
        current_makespan = max(r.last_t for r in self.robots)
        elapsed_s = self._elapsed_plan_seconds()
        print(
            f"[solver] planned {completed}/{total_orders} orders, "
            f"current makespan={current_makespan}, dispatches={dispatch_count}, "
            f"runtime={elapsed_s:.1f}s"
        )
        self._log(
            f"progress completed={completed}/{total_orders} "
            f"makespan={current_makespan} dispatches={dispatch_count} "
            f"runtime_s={elapsed_s:.3f}"
        )

    def _elapsed_plan_seconds(self) -> float:
        if self._plan_started_monotonic <= 0:
            return 0.0
        return time.monotonic() - self._plan_started_monotonic

    def _maybe_validate_dispatch_progress_or_raise(
        self, *, completed: int, total_orders: int, dispatch_count: int
    ) -> None:
        interval = self.config.dispatch_validate_every_makespan
        if interval <= 0:
            return

        current_makespan = max((r.last_t for r in self.robots), default=-1)
        if current_makespan < self._next_dispatch_validation_makespan:
            return

        actions_snapshot = self.actions.sorted_actions()
        validator = SubmissionValidator(worklist_text=self._worklist_text_from_state())
        periodic_error: str | None = None
        try:
            for t, rid, action, x, y in actions_snapshot:
                validator.validate_line(f"{t} {rid} {action} {x} {y}")
            # For periodic dispatch checks we only care about legality/collisions.
            # Incomplete fulfillment is expected mid-run and is not an error.
            validator.finalize()
            strict_conflict = self._strict_no_swap_conflict(actions_snapshot)
            if strict_conflict is not None:
                periodic_error = strict_conflict
        except ValidationError as exc:
            periodic_error = str(exc)
        except Exception as exc:
            periodic_error = f"unexpected validator failure {type(exc).__name__}: {exc!r}"

        if periodic_error is not None:
            self._log(
                "periodic_validation_error "
                f"makespan={current_makespan} dispatches={dispatch_count} "
                f"completed={completed}/{total_orders} error={periodic_error}"
            )
            raise RuntimeError(
                "Periodic dispatch validation failed at "
                f"makespan={current_makespan}, dispatches={dispatch_count}, "
                f"completed_orders={completed}/{total_orders}. "
                f"Validator error: {periodic_error}"
            )

        self._log(
            "periodic_validation_ok "
            f"makespan={current_makespan} dispatches={dispatch_count} "
            f"completed={completed}/{total_orders} actions={len(actions_snapshot)}"
        )
        while self._next_dispatch_validation_makespan <= current_makespan:
            self._next_dispatch_validation_makespan += interval

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

    def _build_relocation_plan(self) -> Deque[RelocationJob]:
        """
        Algorithm: weighted lift heuristic over bucket-level SKU frequencies.
        Pattern: plan builder that ranks relocation candidates then decorates them with targets.
        """
        forced_skus = list(dict.fromkeys(self.config.relocation_skus_to_relocate or []))
        bucket_items = self.past_analysis.bucket_items
        bucket_sku_counts = self.past_analysis.bucket_sku_counts

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
            preferred_hotspot = self._primary_sku_anchor_cell(sku)
            if preferred_hotspot is not None:
                hotspot = preferred_hotspot
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
            preferred_hotspot = self._primary_sku_anchor_cell(sku)
            if preferred_hotspot is not None:
                hotspot = preferred_hotspot
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
        if block_high_use and cell in self.past_analysis.high_use_cells:
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
        sku_rows = self._sku_anchor_rows(sku)
        if not sku_rows:
            return []
        edge_band = self.config.relocation_edge_band
        if edge_band >= 0:
            edge_rows = [row for row in sku_rows if self._is_within_edge_band((row[0], row[1]), edge_band)]
            if edge_rows:
                return edge_rows[: min(len(edge_rows), limit)]
        return sku_rows[: min(len(sku_rows), limit)]

    def _relocate_chunk_size(self) -> int:
        raw = getattr(self.config, "relocate_chunk_size", 1)
        try:
            size = int(raw)
        except (TypeError, ValueError):
            return 1
        return max(1, size)

    def _sku_anchor_rows(self, sku: int) -> List[Tuple[int, int, int]]:
        cache = getattr(self, "_sku_anchor_rows_cache", None)
        if cache is None:
            cache = {}
            self._sku_anchor_rows_cache = cache
        cached = cache.get(sku)
        if cached is not None:
            return cached

        sku_rows = list(self.past_analysis.sku_cells.get(sku, []))
        if not sku_rows:
            cache[sku] = []
            return []

        chunk_size = self._relocate_chunk_size()
        if chunk_size <= 1:
            cache[sku] = sku_rows
            return sku_rows

        chunk_accumulator: Dict[Tuple[int, int], List[float]] = {}
        for x, y, count in sku_rows:
            use_count = int(count)
            if use_count <= 0:
                continue
            key = (int(x) // chunk_size, int(y) // chunk_size)
            current = chunk_accumulator.get(key)
            if current is None:
                current = [0.0, 0.0, 0.0]
            current[0] += use_count
            current[1] += int(x) * use_count
            current[2] += int(y) * use_count
            chunk_accumulator[key] = current

        aggregated_rows: List[Tuple[int, int, int]] = []
        for (chunk_x, chunk_y), (count_sum, weighted_x, weighted_y) in chunk_accumulator.items():
            if count_sum <= 0:
                continue
            anchor_x = int(round(weighted_x / count_sum))
            anchor_y = int(round(weighted_y / count_sum))
            min_x = chunk_x * chunk_size
            min_y = chunk_y * chunk_size
            max_x = min(self.state.width - 1, min_x + chunk_size - 1)
            max_y = min(self.state.height - 1, min_y + chunk_size - 1)
            anchor_x = min(max(anchor_x, min_x), max_x)
            anchor_y = min(max(anchor_y, min_y), max_y)
            aggregated_rows.append((anchor_x, anchor_y, int(count_sum)))

        aggregated_rows.sort(
            key=lambda row: (
                -row[2],
                self.past_analysis.use_by_cell.get((row[0], row[1]), 0),
                row[1],
                row[0],
            )
        )
        cache[sku] = aggregated_rows
        return aggregated_rows

    def _primary_sku_anchor_cell(self, sku: int) -> Tuple[int, int] | None:
        rows = self._sku_anchor_rows(sku)
        if not rows:
            return None
        return rows[0][0], rows[0][1]

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
        edge_band = self.config.relocation_edge_band
        for sx, sy, sku_count in anchor_rows:
            for tx, ty, radius in self._iter_manhattan_cells(sx, sy, max_radius=9):
                cell = (tx, ty)
                if edge_band >= 0 and not self._is_within_edge_band(cell, edge_band):
                    continue
                if not self._is_relocation_target_cell_allowed(
                    cell,
                    reserved_targets=reserved_targets,
                    block_high_use=True,
                ):
                    continue
                use_score = self.past_analysis.use_by_cell.get(cell, 0)
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
                if edge_band >= 0 and not self._is_within_edge_band(cell, edge_band):
                    continue
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

        edge_band = self.config.relocation_edge_band
        for sx, sy, _ in anchor_rows:
            for tx, ty, _ in self._iter_manhattan_cells(sx, sy, max_radius=4):
                cell = (tx, ty)
                if cell in seen:
                    continue
                if edge_band >= 0 and not self._is_within_edge_band(cell, edge_band):
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

    def _is_within_edge_band(self, cell: Tuple[int, int], band: int) -> bool:
        if band < 0:
            return True
        x, y = cell
        edge_dist = min(x, self.state.width - 1 - x, y, self.state.height - 1 - y)
        return edge_dist <= band

    def _suggestion_key(self, suggestion: Suggestion) -> str:
        if isinstance(suggestion, OrderSuggestion):
            return f"order:{suggestion.order_idx}"
        if isinstance(suggestion, SetupSuggestion):
            return f"setup:{suggestion.job.source_pallet_id}"
        if isinstance(suggestion, RelocateSuggestion):
            return f"relocate:{suggestion.job.sku}"
        if isinstance(suggestion, DockSuggestion):
            return f"dock:{suggestion.sku}:{suggestion.center[0]}:{suggestion.center[1]}"
        return f"unknown:{id(suggestion)}"

    def _suggestion_backoff_cycles(self, failure_count: int) -> int:
        base = max(1, int(self.config.suggestion_backoff_base_cycles))
        max_backoff = max(base, int(self.config.suggestion_backoff_max_cycles))
        exp = max(0, int(failure_count) - 1)
        return min(max_backoff, base * (2**exp))

    def _candidate_robots_for_suggestion(self, suggestion: Suggestion) -> List[RobotState]:
        center = suggestion.center

        if isinstance(suggestion, SetupSuggestion):
            setup_robot_by_hotspot = getattr(self, "_setup_robot_by_hotspot", {})
            assigned_robot_id = setup_robot_by_hotspot.get(
                (int(suggestion.job.hotspot[0]), int(suggestion.job.hotspot[1]))
            )
            if assigned_robot_id is not None:
                for robot in self.robots:
                    if robot.id == assigned_robot_id:
                        return [robot]
            ranked = sorted(
                self.robots,
                key=lambda r: self._setup_probe_robot_for_job(r, suggestion.job),
            )
            limit = max(1, min(int(self.config.max_robots_per_suggestion), len(ranked)))
            return ranked[:limit]

        robot_pool = list(self.robots)
        non_setup_robots = [r for r in self.robots if not self._has_pending_setup_for_robot(r.id)]
        if non_setup_robots:
            robot_pool = non_setup_robots

        ranked = sorted(
            robot_pool,
            key=lambda r: (
                r.last_t,
                abs(r.x - center[0]) + abs(r.y - center[1]),
                r.id,
            ),
        )
        limit = max(1, min(int(self.config.max_robots_per_suggestion), len(ranked)))
        return ranked[:limit]

    def _setup_probe_robot_for_job(
        self,
        robot: RobotState,
        job: SetupJob,
    ) -> Tuple[int, int, int, int]:
        """
        Reachability-first ranking key for Setup jobs.
        Returns a tuple ordered as:
        (unreachable_flag, stand_path_len_or_fallback, robot_last_t, robot_id)
        """
        pallet_info = self.pallet_by_id.get(job.source_pallet_id)
        if pallet_info is None:
            dist = abs(robot.x - job.source_xy[0]) + abs(robot.y - job.source_xy[1])
            return (1, dist, robot.last_t, robot.id)

        source_xy = (int(pallet_info["x"]), int(pallet_info["y"]))
        stand_cells = self.scheduler.pick_cells_for_pallet(source_xy)
        if not stand_cells:
            dist = abs(robot.x - source_xy[0]) + abs(robot.y - source_xy[1])
            return (1, dist, robot.last_t, robot.id)

        probe_limit = max(8, min(int(self.config.path_step_limit), 96))
        best_len: int | None = None
        for sx, sy in stand_cells:
            path = self.planner.plan_path(
                robot,
                sx,
                sy,
                max_path_steps=probe_limit,
            )
            if path or (robot.x == sx and robot.y == sy):
                path_len = len(path)
                if best_len is None or path_len < best_len:
                    best_len = path_len

        if best_len is not None:
            return (0, best_len, robot.last_t, robot.id)

        dist = min(abs(robot.x - sx) + abs(robot.y - sy) for sx, sy in stand_cells)
        return (1, dist, robot.last_t, robot.id)

    def _parking_candidate_cells_for_robot(self, robot: RobotState) -> List[Tuple[int, int]]:
        other_robot_cells = {(r.x, r.y) for r in self.robots if r.id != robot.id}
        scored: List[Tuple[int, int, int, int]] = []
        for y in range(self.state.height):
            for x in range(self.state.width):
                cell = (x, y)
                if cell in self.scheduler.pallets:
                    continue
                if cell in self.travel_lane_cells:
                    continue
                if cell in other_robot_cells:
                    continue
                # Avoid blocking perimeter fulfillment cells.
                if x == 0 or x == self.state.width - 1 or y == 0 or y == self.state.height - 1:
                    continue
                dist = abs(x - robot.x) + abs(y - robot.y)
                if dist <= 0:
                    continue
                use_score = int(self.past_analysis.use_by_cell.get(cell, 0))
                scored.append((use_score, dist, y, x))
        scored.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
        limit = max(1, int(self.config.parking_candidate_limit))
        return [(x, y) for _, _, y, x in scored[:limit]]

    def _plan_idle_parking_move(self, robot: RobotState) -> bool:
        if robot.docks:
            return False

        for tx, ty in self._parking_candidate_cells_for_robot(robot):
            path = self._safe_plan_path(robot, tx, ty)
            if not path:
                continue
            if len(path) > self.config.path_step_limit:
                continue

            temp_robot = self._clone_robot_state(robot)
            pending_actions: List[Tuple[int, int, str, int, int]] = []
            pending_paths: List[Tuple[RobotState, List[Tuple[int, int, int]]]] = []
            pending_footprints: List[Tuple[RobotState, int, int, int]] = []

            pending_paths.append((self._clone_robot_state(temp_robot), path))
            pending_actions.extend(self._apply_moves_to_robot(temp_robot, path))
            if not pending_actions:
                continue
            if not self._can_commit_pending_actions(pending_actions):
                continue

            self._commit_plan(
                robot=robot,
                temp_robot=temp_robot,
                pending_actions=pending_actions,
                pending_paths=pending_paths,
                pending_footprints=pending_footprints,
            )
            self._log(
                f"idle_parking_move robot={robot.id} target=({tx},{ty}) "
                f"path_len={len(path)} last_t={robot.last_t}"
            )
            return True
        return False

    def _next_available_robot(self) -> RobotState:
        return min(self.robots, key=lambda r: (r.last_t, r.id))

    def _closest_bucket_for_cell(self, x: int, y: int) -> str:
        choices = list(BUCKET_TO_HOTSPOT.items())
        _, bucket = min(
            ((abs(hx - x) + abs(hy - y), b) for b, (hx, hy) in choices),
            key=lambda row: (row[0], row[1]),
        )
        return bucket

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

        if not self._can_commit_pending_actions(pending_actions):
            return False

        self._commit_plan(
            robot=robot,
            temp_robot=temp_robot,
            pending_actions=pending_actions,
            pending_paths=pending_paths,
            pending_footprints=pending_footprints,
        )
        return True

    def _plan_dock_pallet(self, robot: RobotState, sku: int) -> bool:
        """Plans the actions for a robot to find, travel to, and dock a pallet with a given SKU."""
        source = self._select_relocation_source_pallet(robot, sku)
        if source is None:
            self._log(f"dock_suggestion_fail robot={robot.id} sku={sku} reason=no_source_pallet")
            return False
        pallet_xy, pallet_id = source

        stand_cells = self._candidate_relocation_stand_cells(robot, pallet_xy)
        if not stand_cells:
            self._log(f"dock_suggestion_fail robot={robot.id} sku={sku} reason=no_stand_cells")
            return False

        # Find the first reachable stand cell
        path_to_stand = None
        stand_xy = None
        for cell in stand_cells:
            path = self._safe_plan_path(robot, cell[0], cell[1])
            if path or (robot.x == cell[0] and robot.y == cell[1]):
                path_to_stand = path
                stand_xy = cell
                break

        if stand_xy is None:
            self._log(f"dock_suggestion_fail robot={robot.id} sku={sku} reason=no_path_to_stand")
            return False

        temp_robot = self._clone_robot_state(robot)
        pending_actions: List[Tuple[int, int, str, int, int]] = []
        pending_paths: List[Tuple[RobotState, List[Tuple[int, int, int]]]] = []
        pending_footprints: List[Tuple[RobotState, int, int, int]] = []

        if path_to_stand:
            pending_paths.append((self._clone_robot_state(temp_robot), path_to_stand))
        pending_actions.extend(self._apply_moves_to_robot(temp_robot, path_to_stand))

        dx = pallet_xy[0] - temp_robot.x
        dy = pallet_xy[1] - temp_robot.y

        dock_t = temp_robot.last_t + 1
        pending_actions.append((dock_t, temp_robot.id, "dock", pallet_xy[0], pallet_xy[1]))
        pending_footprints.append((self._clone_robot_state(temp_robot), dock_t, temp_robot.x, temp_robot.y))
        temp_robot.last_t = dock_t
        temp_robot.docks[(dx, dy)] = pallet_id

        if not self._can_commit_pending_actions(pending_actions):
            return False

        self._commit_plan(robot, temp_robot, pending_actions, pending_paths, pending_footprints)
        return True

    def _setup_target_candidates(
        self, job: SetupJob, source_xy: Tuple[int, int], limit: int = 12
    ) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        seen: set[Tuple[int, int]] = set()

        def push(cell: Tuple[int, int]) -> None:
            if cell in seen:
                return
            if cell in self.scheduler.pallets and cell != source_xy:
                return
            seen.add(cell)
            out.append(cell)

        # Strict setup packing: keep each job on its pre-planned target cell.
        # This prevents opportunistic fallback from creating jagged one-wide growth.
        push(job.target_xy)
        return out

    def _setup_source_candidates_for_job(
        self,
        robot: RobotState,
        job: SetupJob,
        *,
        limit: int = 8,
    ) -> List[Tuple[int, Tuple[int, int]]]:
        blocked_source_pallet_ids: set[int] = set()
        for other in self.setup_jobs:
            other_id = int(other.source_pallet_id)
            if other_id == int(job.source_pallet_id):
                continue
            if other_id in self._completed_setup_pallet_ids:
                continue
            if other_id in self._dropped_setup_pallet_ids:
                continue
            blocked_source_pallet_ids.add(other_id)

        rows: List[Tuple[int, int, int, int, Tuple[int, int]]] = []
        seen: set[int] = set()
        hx, hy = int(job.hotspot[0]), int(job.hotspot[1])
        for source_xy in self.scheduler.pallet_cells_for_sku(int(job.sku)):
            source_pallet_id = self.pallet_id_by_coord.get(source_xy)
            if source_pallet_id is None:
                continue
            source_pallet_id = int(source_pallet_id)
            if source_pallet_id in seen:
                continue
            if source_pallet_id in blocked_source_pallet_ids:
                continue
            seen.add(source_pallet_id)
            rows.append(
                (
                    0 if source_pallet_id == int(job.source_pallet_id) else 1,
                    abs(robot.x - source_xy[0]) + abs(robot.y - source_xy[1]),
                    abs(hx - source_xy[0]) + abs(hy - source_xy[1]),
                    source_pallet_id,
                    (int(source_xy[0]), int(source_xy[1])),
                )
            )

        rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
        capped = rows[: max(1, int(limit))]
        return [(source_pallet_id, source_xy) for _, _, _, source_pallet_id, source_xy in capped]

    def _plan_setup_pallet_for_robot(self, robot: RobotState, job: SetupJob) -> bool:
        source_candidates = self._setup_source_candidates_for_job(robot, job)
        if not source_candidates:
            self._log(
                "setup_attempt_fail "
                f"robot={robot.id} sku={job.sku} hotspot={job.hotspot} "
                f"reason=missing_source_pallet source_pallet_id={job.source_pallet_id}"
            )
            return False

        total_stands_tried = 0
        stand_failures = 0
        sources_tried = 0
        failure_reasons: collections.Counter = collections.Counter()

        for source_pallet_id, source_xy in source_candidates:
            sources_tried += 1
            source_xy = (int(source_xy[0]), int(source_xy[1]))
            if source_xy == (int(job.target_xy[0]), int(job.target_xy[1])):
                job.source_pallet_id = int(source_pallet_id)
                job.source_xy = source_xy
                return True

            stand_cells = self._candidate_relocation_stand_cells(robot, source_xy)
            if not stand_cells:
                failure_reasons["no_stand_cells"] += 1
                continue

            target_cells = self._setup_target_candidates(job, source_xy)
            if not target_cells:
                target_blocked = job.target_xy in self.scheduler.pallets and job.target_xy != source_xy
                self._log(
                    "setup_attempt_fail "
                    f"robot={robot.id} sku={job.sku} hotspot={job.hotspot} "
                    f"source={source_xy} target={job.target_xy} "
                    f"reason=no_target_cells target_blocked={target_blocked}"
                )
                return False

            for stand_xy in stand_cells:
                total_stands_tried += 1
                stand_attempt_reasons: collections.Counter = collections.Counter()
                outcome = self._attempt_relocation_via_stand(
                    robot=robot,
                    pallet_xy=source_xy,
                    pallet_id=source_pallet_id,
                    stand_xy=stand_xy,
                    target_pallet_cells=target_cells,
                    setup_redock_edge_step=self._setup_inward_step(job.hotspot),
                    debug_reasons=stand_attempt_reasons,
                )
                if outcome is None:
                    stand_failures += 1
                    failure_reasons.update(stand_attempt_reasons)
                    continue
                new_xy, dock_t, undock_t = outcome
                self._finalize_relocation_pallet_state(
                    pallet_id=source_pallet_id,
                    old_xy=source_xy,
                    new_xy=new_xy,
                    dock_t=dock_t,
                    undock_t=undock_t,
                )
                old_source_pallet_id = int(job.source_pallet_id)
                if int(source_pallet_id) != old_source_pallet_id:
                    self._log(
                        "setup_source_swap "
                        f"hotspot={job.hotspot} sku={job.sku} "
                        f"old_source_pallet_id={old_source_pallet_id} new_source_pallet_id={source_pallet_id}"
                    )
                    self._setup_job_by_source_pallet_id.pop(old_source_pallet_id, None)
                    self._rebind_setup_source_pallet_id(
                        job=job,
                        old_source_pallet_id=old_source_pallet_id,
                        new_source_pallet_id=int(source_pallet_id),
                    )
                job.source_pallet_id = int(source_pallet_id)
                job.source_xy = source_xy
                job.target_xy = new_xy
                self._setup_job_by_source_pallet_id[int(source_pallet_id)] = job
                return True

        self._log(
            "setup_attempt_fail "
            f"robot={robot.id} sku={job.sku} hotspot={job.hotspot} "
            f"source={job.source_xy} target={job.target_xy} "
            f"reason=no_feasible_stand_path sources_tried={sources_tried} "
            f"stands_tried={total_stands_tried} stand_failures={stand_failures} "
            f"failure_reasons={self._format_reason_counts(failure_reasons)}"
        )
        return False

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
        setup_redock_edge_step: int | None = None,
        debug_reasons: collections.Counter | None = None,
    ) -> Tuple[Tuple[int, int], int, int] | None:
        def note(reason: str) -> None:
            if debug_reasons is not None:
                debug_reasons[reason] += 1

        stand_x, stand_y = stand_xy
        temp_robot = self._clone_robot_state(robot)
        pending_actions: List[Tuple[int, int, str, int, int]] = []
        pending_paths: List[Tuple[RobotState, List[Tuple[int, int, int]]]] = []
        pending_footprints: List[Tuple[RobotState, int, int, int]] = []
        pending_static_additions: List[Tuple[int, int, int]] = []

        path_to_stand = self._safe_plan_path(temp_robot, stand_x, stand_y)
        if not path_to_stand and (temp_robot.x != stand_x or temp_robot.y != stand_y):
            note("no_path_to_stand")
            return None
        if path_to_stand:
            pending_paths.append((self._clone_robot_state(temp_robot), path_to_stand))
        pending_actions.extend(self._apply_moves_to_robot(temp_robot, path_to_stand))

        dx = pallet_xy[0] - temp_robot.x
        dy = pallet_xy[1] - temp_robot.y
        if abs(dx) + abs(dy) != 1:
            note("stand_not_adjacent_to_source")
            return None

        dock_t = temp_robot.last_t + 1
        if not self.planner.can_occupy(temp_robot, dock_t, temp_robot.x, temp_robot.y):
            note("dock_footprint_blocked")
            return None
        pending_actions.append((dock_t, temp_robot.id, "dock", pallet_xy[0], pallet_xy[1]))
        pending_footprints.append((self._clone_robot_state(temp_robot), dock_t, temp_robot.x, temp_robot.y))
        temp_robot.last_t = dock_t
        temp_robot.docks[(dx, dy)] = pallet_id

        def try_carry_with_offset(
            carry_robot: RobotState,
            carry_offset: Tuple[int, int],
            carry_pallet_xy: Tuple[int, int],
            target_xy: Tuple[int, int],
        ) -> Tuple[List[Tuple[int, int, int]] | None, str | None]:
            tx2, ty2 = target_xy
            target_robot_x = tx2 - carry_offset[0]
            target_robot_y = ty2 - carry_offset[1]
            if not (0 <= target_robot_x < self.state.width and 0 <= target_robot_y < self.state.height):
                return None, "carry_target_robot_oob"
            if (target_robot_x, target_robot_y) in self.scheduler.pallets and (
                target_robot_x, target_robot_y
            ) != carry_pallet_xy:
                return None, "carry_target_robot_blocked"

            carry_path2 = self._safe_plan_path(carry_robot, target_robot_x, target_robot_y)
            if not carry_path2 and (carry_robot.x != target_robot_x or carry_robot.y != target_robot_y):
                return None, "carry_path_fail"

            candidate_arrival_t = carry_path2[-1][0] if carry_path2 else carry_robot.last_t
            candidate_undock_t = candidate_arrival_t + 1
            candidate_static_from_t = candidate_undock_t + 1
            if not self.planner.can_add_static_obstacle_from(candidate_static_from_t, tx2, ty2):
                return None, "carry_static_obstacle_conflict"
            return carry_path2, None

        def ordered_reorientation_offsets(
            *,
            current_offset: Tuple[int, int],
            preferred_offset: Tuple[int, int] | None,
            target_xy: Tuple[int, int],
            pallet_xy_now: Tuple[int, int],
            robot_anchor_xy: Tuple[int, int],
        ) -> List[Tuple[int, int]]:
            offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            out: List[Tuple[int, int]] = []
            seen: set[Tuple[int, int]] = set()

            def add_offset(offset: Tuple[int, int]) -> None:
                ox, oy = offset
                if (ox, oy) == current_offset:
                    return
                if (ox, oy) in seen:
                    return
                rx = pallet_xy_now[0] - ox
                ry = pallet_xy_now[1] - oy
                if not (0 <= rx < self.state.width and 0 <= ry < self.state.height):
                    return
                if (rx, ry) in self.scheduler.pallets and (rx, ry) != robot_anchor_xy:
                    return
                seen.add((ox, oy))
                out.append((ox, oy))

            if preferred_offset is not None:
                add_offset(preferred_offset)
            heuristic = sorted(
                offsets,
                key=lambda o: (
                    abs((pallet_xy_now[0] - o[0]) - target_xy[0]) + abs((pallet_xy_now[1] - o[1]) - target_xy[1]),
                    o[1],
                    o[0],
                ),
            )
            for offset in heuristic:
                add_offset(offset)
            return out

        chosen_target: Tuple[int, int, List[Tuple[int, int, int]], int] | None = None
        for tx, ty in target_pallet_cells:
            if (tx, ty) in self.scheduler.pallets and (tx, ty) != pallet_xy:
                note("target_cell_occupied")
                continue

            candidate_robot = self._clone_robot_state(temp_robot)
            candidate_paths: List[Tuple[RobotState, List[Tuple[int, int, int]]]] = []
            candidate_actions: List[Tuple[int, int, str, int, int]] = []
            candidate_footprints: List[Tuple[RobotState, int, int, int]] = []
            candidate_offset = (dx, dy)
            candidate_pallet_xy = pallet_xy

            preferred_setup_offset: Tuple[int, int] | None = None
            if setup_redock_edge_step is not None:
                if hasattr(self, "task_planner"):
                    macros = self.task_planner.build_setup_relocation_macros(
                        stand_xy=(stand_x, stand_y),
                        source_xy=pallet_xy,
                        target_xy=(tx, ty),
                        requires_local_maneuver=True,
                    )
                    macro_names = ",".join(m.name for m in macros)
                    self._log(
                        "task_plan_setup_relocation "
                        f"robot={robot.id} source={pallet_xy} target=({tx},{ty}) macros={macro_names}"
                    )
                preferred_setup_offset = (0, -setup_redock_edge_step)

                # Retry redock staging by pulling farther toward target before trying edge-side orientation.
                # This avoids getting stuck when the first reorientation point is blocked.
                redock_done = False
                max_pull_rounds = 3
                base_robot = self._clone_robot_state(candidate_robot)
                base_offset = candidate_offset
                base_pallet_xy = candidate_pallet_xy

                for pull_steps in range(1, max_pull_rounds + 1):
                    staged_robot = self._clone_robot_state(base_robot)
                    staged_paths: List[Tuple[RobotState, List[Tuple[int, int, int]]]] = []
                    staged_actions: List[Tuple[int, int, str, int, int]] = []
                    staged_footprints: List[Tuple[RobotState, int, int, int]] = []
                    staged_offset = base_offset
                    staged_pallet_xy = base_pallet_xy

                    staged_ok = True
                    for _ in range(pull_steps):
                        step_moved = False
                        for move_dx, move_dy in self._setup_pull_directions(staged_pallet_xy, (tx, ty)):
                            pull_robot_x = staged_robot.x + move_dx
                            pull_robot_y = staged_robot.y + move_dy
                            pull_pallet_x = staged_pallet_xy[0] + move_dx
                            pull_pallet_y = staged_pallet_xy[1] + move_dy
                            if not (0 <= pull_robot_x < self.state.width and 0 <= pull_robot_y < self.state.height):
                                note("redock_pull_robot_oob")
                                continue
                            if not (0 <= pull_pallet_x < self.state.width and 0 <= pull_pallet_y < self.state.height):
                                note("redock_pull_pallet_oob")
                                continue
                            pulled_xy = (pull_pallet_x, pull_pallet_y)
                            if pulled_xy in self.scheduler.pallets and pulled_xy != staged_pallet_xy:
                                note("redock_pull_target_blocked")
                                continue

                            pull_path = self._safe_plan_path_with_step_cap(
                                staged_robot,
                                pull_robot_x,
                                pull_robot_y,
                                max_path_steps=1,
                            )
                            if not pull_path and (staged_robot.x != pull_robot_x or staged_robot.y != pull_robot_y):
                                note("redock_pull_path_fail")
                                continue
                            if pull_path:
                                staged_paths.append((self._clone_robot_state(staged_robot), pull_path))
                            staged_actions.extend(self._apply_moves_to_robot(staged_robot, pull_path))
                            staged_pallet_xy = pulled_xy
                            step_moved = True
                            break
                        if not step_moved:
                            staged_ok = False
                            break

                    if not staged_ok:
                        note("redock_pull_step_failed")
                        continue

                    # Motion planner (local 5x5 mini-game) handles the micro maneuver:
                    # undock -> local moves -> dock from desired side.
                    target_offset = preferred_setup_offset
                    if target_offset is None:
                        note("redock_target_offset_missing")
                        continue
                    redock_ok, staged_offset = self._execute_local_pivot_maneuver(
                        staged_robot=staged_robot,
                        pallet_id=pallet_id,
                        staged_pallet_xy=staged_pallet_xy,
                        staged_offset=staged_offset,
                        target_offset=target_offset,
                        staged_paths=staged_paths,
                        staged_actions=staged_actions,
                        staged_footprints=staged_footprints,
                        note=note,
                    )
                    if not redock_ok:
                        continue

                    candidate_robot = staged_robot
                    candidate_paths.extend(staged_paths)
                    candidate_actions.extend(staged_actions)
                    candidate_footprints.extend(staged_footprints)
                    candidate_offset = staged_offset
                    candidate_pallet_xy = staged_pallet_xy
                    redock_done = True
                    break

                if not redock_done:
                    note("redock_not_possible")
                    # Fall through into generalized maneuver attempts as a backup.

            carry_path, carry_reason = try_carry_with_offset(
                candidate_robot,
                candidate_offset,
                candidate_pallet_xy,
                (tx, ty),
            )

            if carry_path is None:
                if carry_reason is not None:
                    note(carry_reason)
                if hasattr(self, "task_planner"):
                    macros = self.task_planner.build_setup_relocation_macros(
                        stand_xy=(stand_x, stand_y),
                        source_xy=pallet_xy,
                        target_xy=(tx, ty),
                        requires_local_maneuver=True,
                    )
                    macro_names = ",".join(m.name for m in macros)
                    self._log(
                        "task_plan_relocation_fallback "
                        f"robot={robot.id} source={pallet_xy} target=({tx},{ty}) macros={macro_names}"
                    )

                reorientation_offsets = ordered_reorientation_offsets(
                    current_offset=candidate_offset,
                    preferred_offset=preferred_setup_offset,
                    target_xy=(tx, ty),
                    pallet_xy_now=candidate_pallet_xy,
                    robot_anchor_xy=(candidate_robot.x, candidate_robot.y),
                )
                for target_offset in reorientation_offsets:
                    trial_robot = self._clone_robot_state(candidate_robot)
                    trial_paths = list(candidate_paths)
                    trial_actions = list(candidate_actions)
                    trial_footprints = list(candidate_footprints)

                    reorient_ok, new_offset = self._execute_local_pivot_maneuver(
                        staged_robot=trial_robot,
                        pallet_id=pallet_id,
                        staged_pallet_xy=candidate_pallet_xy,
                        staged_offset=candidate_offset,
                        target_offset=target_offset,
                        staged_paths=trial_paths,
                        staged_actions=trial_actions,
                        staged_footprints=trial_footprints,
                        note=note,
                    )
                    if not reorient_ok:
                        continue

                    trial_carry_path, trial_reason = try_carry_with_offset(
                        trial_robot,
                        new_offset,
                        candidate_pallet_xy,
                        (tx, ty),
                    )
                    if trial_carry_path is None:
                        if trial_reason is not None:
                            note(f"{trial_reason}_after_reorient")
                        continue

                    candidate_robot = trial_robot
                    candidate_paths = trial_paths
                    candidate_actions = trial_actions
                    candidate_footprints = trial_footprints
                    candidate_offset = new_offset
                    carry_path = trial_carry_path
                    break

            if carry_path is None:
                continue

            pending_paths.extend(candidate_paths)
            pending_actions.extend(candidate_actions)
            pending_footprints.extend(candidate_footprints)
            temp_robot = candidate_robot
            dx, dy = candidate_offset
            chosen_target = (tx, ty, carry_path)
            break

        if chosen_target is None:
            note("no_target_candidate_after_stand")
            return None

        target_pallet_x, target_pallet_y, carry_path = chosen_target
        if carry_path:
            pending_paths.append((self._clone_robot_state(temp_robot), carry_path))
        pending_actions.extend(self._apply_moves_to_robot(temp_robot, carry_path))

        undock_t = temp_robot.last_t + 1
        if not self.planner.can_occupy(temp_robot, undock_t, temp_robot.x, temp_robot.y):
            note("final_undock_blocked")
            return None
        pending_actions.append((undock_t, temp_robot.id, "undock", target_pallet_x, target_pallet_y))
        pending_footprints.append((self._clone_robot_state(temp_robot), undock_t, temp_robot.x, temp_robot.y))
        temp_robot.last_t = undock_t
        temp_robot.docks.pop((dx, dy), None)

        pending_static_additions.append((undock_t + 1, target_pallet_x, target_pallet_y))
        if not self._can_commit_pending_actions(pending_actions):
            note("candidate_validation_rejected")
            return None
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
            try:
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
            except Exception as exc:
                self._log(
                    "relocation_attempt_error "
                    f"robot={robot.id} sku={job.sku} pallet_id={pallet_id} "
                    f"stand={stand_xy} source={pallet_xy} error={type(exc).__name__}: {exc!r}"
                )
                continue

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
        for x, y in self.past_analysis.high_use_cells:
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
        start_x, start_y = robot.x, robot.y
        started = time.perf_counter()
        path = self.planner.plan_path(
            robot,
            target_x,
            target_y,
            max_path_steps=self.config.path_step_limit,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._astar_calls += 1
        self._astar_total_ms += elapsed_ms
        if elapsed_ms > self._astar_max_ms:
            self._astar_max_ms = elapsed_ms

        blocked = not path and (start_x != target_x or start_y != target_y)
        if blocked:
            self._astar_blocked_calls += 1

        if elapsed_ms >= self.config.astar_slow_ms:
            self._astar_slow_calls += 1
            msg = (
                "astar_slow "
                f"robot={robot.id} start=({start_x},{start_y}) target=({target_x},{target_y}) "
                f"last_t={robot.last_t} docks={len(robot.docks)} path_len={len(path)} "
                f"blocked={blocked} elapsed_ms={elapsed_ms:.3f}"
            )
            self._log(msg)
            if self.config.astar_print_slow:
                print(f"[solver] {msg}")
        elif blocked and self.config.astar_log_blocked:
            self._log(
                "astar_blocked "
                f"robot={robot.id} start=({start_x},{start_y}) target=({target_x},{target_y}) "
                f"last_t={robot.last_t} docks={len(robot.docks)} elapsed_ms={elapsed_ms:.3f}"
            )
        return path

    def _safe_plan_path_with_step_cap(
        self,
        robot: RobotState,
        target_x: int,
        target_y: int,
        *,
        max_path_steps: int,
    ) -> List[Tuple[int, int, int]]:
        """
        Fast bounded path query for short staging maneuvers.
        Keeps setup redock attempts from spending long A* searches on impossible micro-moves.
        """
        start_x, start_y = robot.x, robot.y
        started = time.perf_counter()
        path = self.planner.plan_path(
            robot,
            target_x,
            target_y,
            max_path_steps=max(0, int(max_path_steps)),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._astar_calls += 1
        self._astar_total_ms += elapsed_ms
        if elapsed_ms > self._astar_max_ms:
            self._astar_max_ms = elapsed_ms

        blocked = not path and (start_x != target_x or start_y != target_y)
        if blocked:
            self._astar_blocked_calls += 1

        if elapsed_ms >= self.config.astar_slow_ms:
            self._astar_slow_calls += 1
            msg = (
                "astar_slow "
                f"robot={robot.id} start=({start_x},{start_y}) target=({target_x},{target_y}) "
                f"last_t={robot.last_t} docks={len(robot.docks)} path_len={len(path)} "
                f"blocked={blocked} elapsed_ms={elapsed_ms:.3f}"
            )
            self._log(msg)
            if self.config.astar_print_slow:
                print(f"[solver] {msg}")
        elif blocked and self.config.astar_log_blocked:
            self._log(
                "astar_blocked "
                f"robot={robot.id} start=({start_x},{start_y}) target=({target_x},{target_y}) "
                f"last_t={robot.last_t} docks={len(robot.docks)} elapsed_ms={elapsed_ms:.3f}"
            )
        return path

    def _can_commit_pending_actions(
        self, pending_actions: List[Tuple[int, int, str, int, int]]
    ) -> bool:
        if not pending_actions:
            return True
        candidate = self.actions.sorted_actions() + list(pending_actions)
        candidate.sort(key=lambda row: (row[0], row[1]))
        if not self._has_unique_robot_timestep_pairs(candidate):
            return False
        return self._validate_candidate_actions(
            candidate,
            log_on_error=True,
            require_complete=False,
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
        for i in range(self.config.lns_iterations):
            if i > 0 and i % 10 == 0:
                print(f"[lns] iteration {i}/{self.config.lns_iterations}, best makespan={best_makespan}")
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
        require_complete: bool = True,
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
        strict_conflict = self._strict_no_swap_conflict(actions)
        if strict_conflict is not None:
            if log_on_error:
                self._log(f"candidate_validation_error: {strict_conflict}")
            return False
        if not require_complete:
            return True
        ok = final_state.fulfilled_orders == final_state.total_orders
        if log_on_error and not ok:
            self._log(
                f"candidate_validation_incomplete fulfilled={final_state.fulfilled_orders} "
                f"total={final_state.total_orders} next_t={final_state.next_timestep}"
            )
        return ok

    def _strict_no_swap_conflict(
        self, actions: List[Tuple[int, int, str, int, int]]
    ) -> str | None:
        if not self.config.strict_no_swap:
            return None
        if not actions:
            return None

        by_t: Dict[int, Dict[int, Tuple[str, int, int]]] = collections.defaultdict(dict)
        for t, rid, action, x, y in actions:
            by_t[t][rid] = (action, x, y)

        positions = {rid: (x, y) for rid, (x, y) in enumerate(self.state.robots)}
        max_t = max(by_t) if by_t else -1
        for t in range(max_t + 1):
            acts = by_t.get(t, {})
            start_positions = dict(positions)
            moving_targets: Dict[int, Tuple[int, int]] = {
                rid: (x, y)
                for rid, (action, x, y) in acts.items()
                if action == "move"
            }
            if moving_targets:
                rids = sorted(moving_targets.keys())
                for idx, rid in enumerate(rids):
                    my_start = start_positions.get(rid)
                    my_target = moving_targets[rid]
                    if my_start is None:
                        continue
                    for other in rids[idx + 1 :]:
                        other_start = start_positions.get(other)
                        other_target = moving_targets[other]
                        if other_start is None:
                            continue
                        if my_target == other_start and other_target == my_start:
                            return (
                                f"timestep {t}: strict_no_swap violation: robots {rid} and {other} "
                                f"would swap ({my_start[0]}, {my_start[1]}) <-> ({other_start[0]}, {other_start[1]})"
                            )
            for rid, (action, x, y) in acts.items():
                if action == "move":
                    positions[rid] = (x, y)
        return None

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
        # Append so solve + optimize phases and multi-run iterations keep a full trace.
        self._log_handle = log_path.open("a", encoding="utf-8", buffering=1)

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _log(self, message: str) -> None:
        log_handle = getattr(self, "_log_handle", None)
        if log_handle is None:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        log_handle.write(f"[{ts}] {message}\n")

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
        pallet_cell_cache: Dict[int, set[Tuple[int, int]]] = {}

        def static_pallet_cells_at(t: int) -> set[Tuple[int, int]]:
            cached = pallet_cell_cache.get(t)
            if cached is not None:
                return cached
            cells: set[Tuple[int, int]] = set()
            for pallet_id in self.pallet_by_id.keys():
                xy = self._pallet_static_xy_at(pallet_id, t)
                if xy is not None:
                    cells.add(xy)
            pallet_cell_cache[t] = cells
            return cells

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
                static_pallet_cells = static_pallet_cells_at(t)
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
                            if candidate_fp.intersection(static_pallet_cells):
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
