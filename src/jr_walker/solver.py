import collections
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Tuple

from jr_walker.logic import EdgeAwareOrderScorer, OrderOptimizer
from jr_walker.planner import adjacent_cells
from jr_walker.planner import ReservationPlanner
from jr_walker.scheduler import GreedyScheduler
from jr_walker.sim import ActionLog, RobotState
from jr_walker.writer import write_actions

ROLE_DELIVER = "deliver"
ROLE_RELOCATE_PALLET = "relocate_pallet"
DELIVER_EASY = "easy"
DELIVER_HARD = "hard"

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
class SolverConfig:
    max_time: int = 50000
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
    log_path: Path | None = None
    dispatch_log_every: int = 1


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
        sku_counter: collections.Counter = collections.Counter()
        for order in self.orders:
            sku_counter.update(order.items)
        self.skus_by_demand: List[int] = [sku for sku, _ in sku_counter.most_common()]
        self.relocation_plan: Deque[RelocationJob] = self._build_relocation_plan()

        self._next_delivery_strategy_by_robot: Dict[int, str] = {
            robot.id: DELIVER_EASY for robot in self.robots
        }
        self._initial_relocate_assigned = 0
        self._dispatch_floor_t = -1
        self._warmup_barrier_applied = False
        self._log_handle = None

    def solve(self) -> Tuple[Path, List[Tuple[int, int, str, int, int]]]:
        self._open_log()
        remaining_orders = self._build_ranked_order_queue()
        self._recalculate_order_costs(remaining_orders)
        total_orders = len(remaining_orders)
        completed = 0
        dispatch_count = 0
        self._log(f"solve_start total_orders={total_orders}")
        if self.relocation_plan:
            summary = ", ".join(
                f"SKU{job.sku}->{job.bucket}@{job.placement_offset}(score={job.score:.2f})"
                for job in list(self.relocation_plan)[:6]
            )
            self._log(f"relocation_plan count={len(self.relocation_plan)} top=[{summary}]")
        else:
            self._log("relocation_plan count=0")

        try:
            while remaining_orders:
                robot = self._next_available_robot()
                role = self._dispatch_role(robot)
                strategy = self._next_delivery_strategy_by_robot.get(robot.id, DELIVER_EASY)
                if role == ROLE_DELIVER:
                    self._ensure_warmup_barrier()

                if robot.last_t < self._dispatch_floor_t:
                    robot.last_t = self._dispatch_floor_t
                dnum = dispatch_count + 1
                if dnum % self.config.dispatch_log_every == 0:
                    role_label = role
                    if role == ROLE_DELIVER:
                        role_label = f"{ROLE_DELIVER}_{strategy}"
                    self._log(
                        f"dispatch_start n={dnum} robot={robot.id} role={role_label} "
                        f"robot_t={robot.last_t} completed={completed}/{total_orders} "
                        f"remaining={len(remaining_orders)}"
                    )

                t0 = time.perf_counter()
                if role == ROLE_DELIVER:
                    handled = self._role_deliver(robot, remaining_orders, strategy=strategy)
                    if handled:
                        self._toggle_delivery_strategy(robot.id)
                else:
                    handled = self._role_relocate_pallet(robot, remaining_orders)
                if not handled and role != ROLE_DELIVER:
                    self._ensure_warmup_barrier()
                    handled = self._deliver_with_robot_strategy(robot, remaining_orders)

                if not handled:
                    handled = self._fallback_deliver_any_robot(remaining_orders)

                elapsed = time.perf_counter() - t0
                if dnum % self.config.dispatch_log_every == 0:
                    self._log(
                        f"dispatch_end n={dnum} robot={robot.id} role={role} success={handled} "
                        f"elapsed_s={elapsed:.2f} completed={total_orders - len(remaining_orders)}/{total_orders} "
                        f"remaining={len(remaining_orders)}"
                    )

                if not handled:
                    raise RuntimeError("Dispatcher could not assign a feasible next task.")

                new_completed = total_orders - len(remaining_orders)
                if new_completed != completed:
                    completed = new_completed
                    if completed % self.config.progress_every == 0:
                        current_makespan = max(r.last_t for r in self.robots)
                        print(
                            f"[solver] planned {completed}/{total_orders} orders, "
                            f"current makespan={current_makespan}, dispatches={dispatch_count + 1}"
                        )
                        self._log(
                            f"progress completed={completed}/{total_orders} "
                            f"makespan={current_makespan} dispatches={dispatch_count + 1}"
                        )

                dispatch_count += 1

            sorted_actions = self.actions.sorted_actions()
            sorted_actions = self._repair_idle_wait_conflicts(sorted_actions)
            output_path = write_actions(sorted_actions, self.config.output_path)
            makespan = max((t for t, _, _, _, _ in sorted_actions), default=-1)
            self._log(
                f"solve_end actions={len(sorted_actions)} makespan={makespan} output={output_path}"
            )
            return output_path, sorted_actions
        finally:
            self._close_log()

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

    def _build_relocation_plan(self) -> Deque[RelocationJob]:
        analysis_path = self.config.relocation_analysis_path
        if analysis_path is None:
            analysis_path = self._find_default_analysis_path()
        if analysis_path is None:
            return collections.deque()
        analysis_path = Path(analysis_path)
        if not analysis_path.exists():
            return collections.deque()

        bucket_items, bucket_sku_counts = self._parse_analysis_file(analysis_path)
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
            jobs.append(
                RelocationJob(
                    sku=sku,
                    bucket=bucket,
                    hotspot=BUCKET_TO_HOTSPOT[bucket],
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
        planned_targets: set[Tuple[int, int]] = set()
        for job in jobs:
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
        return collections.deque(jobs)

    def _find_default_analysis_path(self) -> Path | None:
        output_dir = Path("output")
        if not output_dir.exists():
            return None
        candidates = sorted(output_dir.glob("solution_*_analysis.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return None
        return candidates[0]

    def _parse_analysis_file(
        self, analysis_path: Path
    ) -> Tuple[Dict[str, int], Dict[str, collections.Counter]]:
        bucket_items: Dict[str, int] = {}
        bucket_sku_counts: Dict[str, collections.Counter] = {}
        current_bucket = None

        for raw_line in analysis_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                current_bucket = line[1:-1]
                if current_bucket not in bucket_sku_counts:
                    bucket_sku_counts[current_bucket] = collections.Counter()
                continue

            if current_bucket is None:
                continue

            if line.startswith("items:"):
                try:
                    bucket_items[current_bucket] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    bucket_items[current_bucket] = 0
                continue

            if line.startswith("sku_counts:"):
                value = line.split(":", 1)[1].strip()
                if value == "(none)":
                    continue
                parts = [p.strip() for p in value.split(",") if p.strip()]
                for part in parts:
                    if not part.startswith("SKU") or ":" not in part:
                        continue
                    sku_part, count_part = part.split(":", 1)
                    try:
                        sku = int(sku_part[3:])
                        count = int(count_part)
                    except ValueError:
                        continue
                    bucket_sku_counts[current_bucket][sku] = count

        return bucket_items, bucket_sku_counts

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

    def _dispatch_role(self, robot: RobotState) -> str:
        if (
            self._initial_relocate_assigned < self.config.initial_relocate_dispatches
            and self._has_relocation_candidate()
        ):
            self._initial_relocate_assigned += 1
            return ROLE_RELOCATE_PALLET

        if self._has_relocation_candidate():
            if self.rng.random() < self.config.relocate_pallet_probability:
                return ROLE_RELOCATE_PALLET

        return ROLE_DELIVER

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

    def _role_deliver(self, robot: RobotState, remaining_orders: Deque[int], strategy: str) -> bool:
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

    def _role_relocate_pallet(self, robot: RobotState, remaining_orders: Deque[int]) -> bool:
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

    def _plan_relocate_pallet_for_robot(self, robot: RobotState, job: RelocationJob) -> bool:
        sku = job.sku
        pallet_cells = self.scheduler.pallet_cells_for_sku(sku)
        if not pallet_cells:
            return False

        rx, ry = robot.x, robot.y
        pallet_xy = min(pallet_cells, key=lambda p: abs(p[0] - rx) + abs(p[1] - ry))
        pallet_id = self.pallet_id_by_coord.get(pallet_xy)
        if pallet_id is None:
            return False

        stand_cells = self.scheduler.pick_cells_for_pallet(pallet_xy)
        if not stand_cells:
            return False
        stand_cells.sort(key=lambda p: abs(p[0] - rx) + abs(p[1] - ry))
        stand_cells = stand_cells[: self.config.relocate_stand_candidate_limit]

        target_pallet_cells = self._candidate_relocation_targets(job)
        target_pallet_cells.sort(
            key=lambda p: (
                0 if p == job.preferred_target_xy else 1,
                abs(p[0] - pallet_xy[0]) + abs(p[1] - pallet_xy[1]),
            )
        )
        target_pallet_cells = target_pallet_cells[: self.config.relocate_target_candidate_limit]

        for stand_x, stand_y in stand_cells:
            temp_robot = self._clone_robot_state(robot)
            pending_actions: List[Tuple[int, int, str, int, int]] = []
            pending_paths: List[Tuple[RobotState, List[Tuple[int, int, int]]]] = []
            pending_footprints: List[Tuple[RobotState, int, int, int]] = []
            pending_static_additions: List[Tuple[int, int, int]] = []

            path_to_stand = self._safe_plan_path(temp_robot, stand_x, stand_y)
            if not path_to_stand and (temp_robot.x != stand_x or temp_robot.y != stand_y):
                continue
            if path_to_stand:
                pending_paths.append((self._clone_robot_state(temp_robot), path_to_stand))
            pending_actions.extend(self._apply_moves_to_robot(temp_robot, path_to_stand))

            dx = pallet_xy[0] - temp_robot.x
            dy = pallet_xy[1] - temp_robot.y
            if abs(dx) + abs(dy) != 1:
                continue

            dock_t = temp_robot.last_t + 1
            if not self.planner.can_occupy(temp_robot, dock_t, temp_robot.x, temp_robot.y):
                continue
            pending_actions.append((dock_t, temp_robot.id, "dock", pallet_xy[0], pallet_xy[1]))
            pending_footprints.append((self._clone_robot_state(temp_robot), dock_t, temp_robot.x, temp_robot.y))
            temp_robot.last_t = dock_t
            temp_robot.docks[(dx, dy)] = pallet_id

            chosen = None
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

                chosen = (tx, ty, carry_path)
                break

            if chosen is None:
                continue

            target_pallet_x, target_pallet_y, carry_path = chosen
            if carry_path:
                pending_paths.append((self._clone_robot_state(temp_robot), carry_path))
            pending_actions.extend(self._apply_moves_to_robot(temp_robot, carry_path))

            undock_t = temp_robot.last_t + 1
            if not self.planner.can_occupy(temp_robot, undock_t, temp_robot.x, temp_robot.y):
                continue
            pending_actions.append((undock_t, temp_robot.id, "undock", target_pallet_x, target_pallet_y))
            pending_footprints.append((self._clone_robot_state(temp_robot), undock_t, temp_robot.x, temp_robot.y))
            temp_robot.last_t = undock_t
            del temp_robot.docks[(dx, dy)]

            # The undocked pallet becomes static from the next timestep onward.
            pending_static_additions.append((undock_t + 1, target_pallet_x, target_pallet_y))

            self._commit_plan(
                robot=robot,
                temp_robot=temp_robot,
                pending_actions=pending_actions,
                pending_paths=pending_paths,
                pending_footprints=pending_footprints,
                pending_static_additions=pending_static_additions,
            )

            old_xy = pallet_xy
            new_xy = (target_pallet_x, target_pallet_y)
            if new_xy != old_xy:
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
            if (tx, ty) in reserved_targets:
                continue
            return (dx, dy)

        for dx, dy in candidates:
            tx, ty = hotspot[0] + dx, hotspot[1] + dy
            if 0 <= tx < self.state.width and 0 <= ty < self.state.height:
                fallback = (dx, dy)
                break
        return fallback

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
