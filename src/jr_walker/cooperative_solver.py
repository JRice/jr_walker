from __future__ import annotations

import collections
import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from jr_walker.sim import RobotState
from jr_walker.solver import WarehouseSolver


@dataclass
class NestState:
    nest_id: int
    anchor: Tuple[int, int]
    robot_ids: List[int] = field(default_factory=list)
    remaining_skus: List[int] = field(default_factory=list)
    target_cells: List[Tuple[int, int]] = field(default_factory=list)
    placed_jobs: List[Tuple[int, int, Tuple[int, int]]] = field(default_factory=list)


class CooperativeWarehouseSolver(WarehouseSolver):
    """
    Cooperative dual-nest scheduler:
    - setup is planned per nest in alternating turns
    - robots are permanently assigned to a nest
    - order jobs follow a fixed conveyor loop (A* only for first positioning step)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._next_tick_progress = max(1, int(getattr(self.config, "progress_tick_interval", 1000)))

    def _find_solution_actions_core(self) -> List[Tuple[int, int, str, int, int]]:
        if not bool(getattr(self.config, "single_nest_conveyor_mode", False)):
            raise RuntimeError("Set solver.single_nest_conveyor_mode=true.")

        hotspots = self._single_nest_top_hotspots(
            max(1, len(list(getattr(self.config, "setup_hotspots", []) or [])))
        )
        if len(hotspots) < 2:
            return super()._find_solution_actions_single_nest_conveyor()
        return self._find_solution_actions_cooperative_dual_nest(hotspots)

    def _find_solution_actions_cooperative_dual_nest(
        self, hotspots: List[Tuple[int, int]]
    ) -> List[Tuple[int, int, str, int, int]]:
        self._plan_started_monotonic = time.monotonic()
        total_orders = len(self.orders)
        self._completed_order_indices = set()
        self._log(f"solve_start total_orders={total_orders} mode=cooperative_dual_nest")

        nest_anchors = [(int(h[0]), 0) for h in hotspots]
        if len(self.robots) < len(nest_anchors):
            raise RuntimeError(
                f"Need at least one robot per nest (robots={len(self.robots)} nests={len(nest_anchors)})."
            )

        skus = self._active_skus_for_nest()
        if len(skus) > 20:
            skus = skus[:20]
        if len(skus) < 1:
            raise RuntimeError("No active SKUs found in order list.")

        self._reset_hotspot_masks_for_cooperative_mode()
        nests = self._build_nest_states(nest_anchors=nest_anchors, skus=skus)
        self._assign_robots_to_nests(nests)
        self._plan_cooperative_setup(nests)
        self._plan_order_phase(nests)

        out = self.actions.sorted_actions()
        repaired = self._repair_idle_wait_conflicts(out)
        if repaired != out and self._validate_candidate_actions(repaired, log_on_error=True):
            out = repaired

        makespan = max((t for t, _, _, _, _ in out), default=-1)
        self._log(f"find_solution_end actions={len(out)} makespan={makespan} mode=cooperative_dual_nest")
        return out

    def _reset_hotspot_masks_for_cooperative_mode(self) -> None:
        self._setup_robot_by_hotspot = {}
        self._setup_robot_ids = set()
        self._robot_hotspot_by_id = {}
        self._hotspot_protected_cells_by_hotspot = {}
        self._hotspot_astar_forbidden_cells_by_hotspot = {}
        self._non_hotspot_forbidden_cells = set()
        self._non_hotspot_astar_forbidden_cells = set()
        for robot in self.robots:
            if hasattr(robot, "assigned_hotspot"):
                delattr(robot, "assigned_hotspot")

    def _active_skus_for_nest(self) -> List[int]:
        skus: set[int] = set()
        for order in self.orders:
            skus.update(int(s) for s in order.items.keys())
        if not skus:
            skus = {int(s) for s in self.scheduler.pallets.values()}
        return sorted(skus)

    def _build_nest_states(self, *, nest_anchors: List[Tuple[int, int]], skus: List[int]) -> List[NestState]:
        nests: List[NestState] = []
        for idx, anchor in enumerate(nest_anchors):
            cells = self._nest_target_cells(anchor[0])
            if len(skus) > len(cells):
                raise RuntimeError(
                    f"Nest {idx} has {len(cells)} target cells but needs {len(skus)} SKUs."
                )
            nests.append(
                NestState(
                    nest_id=int(idx),
                    anchor=(int(anchor[0]), 0),
                    remaining_skus=list(skus),
                    target_cells=list(cells),
                )
            )
        return nests

    def _nest_target_cells(self, nest_x: int) -> List[Tuple[int, int]]:
        row0 = [(int(nest_x) + i, 0) for i in range(10)]
        row2 = [(int(nest_x) + i, 2) for i in range(10)]
        return row0 + row2

    def _assign_robots_to_nests(self, nests: List[NestState]) -> None:
        by_id = {int(r.id): r for r in self.robots}
        available = set(by_id.keys())
        while available:
            assigned_this_round = False
            for nest in nests:
                if not available:
                    break
                ax, ay = nest.anchor
                best = min(
                    available,
                    key=lambda rid: (
                        abs(int(by_id[rid].x) - ax) + abs(int(by_id[rid].y) - ay),
                        rid,
                    ),
                )
                nest.robot_ids.append(int(best))
                setattr(by_id[best], "assigned_nest_id", int(nest.nest_id))
                setattr(by_id[best], "assigned_nest_anchor", nest.anchor)
                available.remove(best)
                assigned_this_round = True
            if not assigned_this_round:
                break

        empty = [n.nest_id for n in nests if not n.robot_ids]
        if empty:
            raise RuntimeError(f"Nests without robots: {empty}")

    def _plan_cooperative_setup(self, nests: List[NestState]) -> None:
        planned_pallet_ids: set[int] = set()
        placement_events: List[Tuple[int, int, int, int]] = []

        while any(n.remaining_skus for n in nests):
            remaining_setup = sum(len(n.remaining_skus) for n in nests)
            self._check_global_limits_or_raise(collections.deque(range(remaining_setup)))
            progress = False
            for nest in nests:
                if not nest.remaining_skus:
                    continue
                if not nest.target_cells:
                    raise RuntimeError(f"Nest {nest.nest_id} has no remaining target cells.")
                target_xy = nest.target_cells[0]
                success = self._assign_next_setup_job(
                    nest=nest,
                    target_xy=target_xy,
                    planned_pallet_ids=planned_pallet_ids,
                    placement_events=placement_events,
                )
                if not success:
                    raise RuntimeError(
                        f"Failed setup assignment for nest={nest.nest_id} target={target_xy} "
                        f"remaining_skus={nest.remaining_skus}"
                    )
                progress = True

            if not progress:
                raise RuntimeError("Cooperative setup deadlock: no setup jobs were planned in a full cycle.")

        self._park_robots_for_last_setup_wave(placement_events)
        for nest in nests:
            self._verify_nest_layout(nest)

    def _assign_next_setup_job(
        self,
        *,
        nest: NestState,
        target_xy: Tuple[int, int],
        planned_pallet_ids: set[int],
        placement_events: List[Tuple[int, int, int, int]],
    ) -> bool:
        candidates = self._nearest_unplanned_pallet_candidates(
            sku_pool=nest.remaining_skus,
            target_xy=target_xy,
            planned_pallet_ids=planned_pallet_ids,
            limit=24,
        )
        if not candidates:
            return False

        robots = [self._robot_by_id(rid) for rid in nest.robot_ids]
        for _dist, pallet_id, source_xy, sku in candidates:
            robots_ranked = sorted(
                robots,
                key=lambda r: (
                    int(r.last_t),
                    abs(int(r.x) - int(source_xy[0])) + abs(int(r.y) - int(source_xy[1])),
                    int(r.id),
                ),
            )
            for robot in robots_ranked:
                if robot.docks:
                    self._plan_idle_recovery_undock(robot)
                ok = self._plan_single_nest_pallet_move(
                    robot=robot,
                    pallet_id=int(pallet_id),
                    target_xy=(int(target_xy[0]), int(target_xy[1])),
                    hotspot=nest.anchor,
                )
                if not ok:
                    continue
                planned_pallet_ids.add(int(pallet_id))
                nest.target_cells.pop(0)
                if int(sku) in nest.remaining_skus:
                    nest.remaining_skus.remove(int(sku))
                nest.placed_jobs.append((len(nest.placed_jobs), int(sku), (int(target_xy[0]), int(target_xy[1]))))
                setattr(robot, "last_job_kind", "setup")
                placement_events.append((int(robot.last_t), int(robot.id), int(nest.nest_id), int(pallet_id)))
                return True
        return False

    def _nearest_unplanned_pallet_candidates(
        self,
        *,
        sku_pool: Iterable[int],
        target_xy: Tuple[int, int],
        planned_pallet_ids: set[int],
        limit: int,
    ) -> List[Tuple[int, int, Tuple[int, int], int]]:
        tx, ty = int(target_xy[0]), int(target_xy[1])
        rows: List[Tuple[int, int, Tuple[int, int], int]] = []
        for sku in sorted(int(s) for s in sku_pool):
            for source_xy in self.scheduler.pallet_cells_for_sku(int(sku)):
                source = (int(source_xy[0]), int(source_xy[1]))
                pallet_id = self.pallet_id_by_coord.get(source)
                if pallet_id is None:
                    continue
                pid = int(pallet_id)
                if pid in planned_pallet_ids:
                    continue
                if pid in getattr(self, "_persistently_docked_pallet_ids", set()):
                    continue
                dist = abs(source[0] - tx) + abs(source[1] - ty)
                rows.append((int(dist), pid, source, int(sku)))
        rows.sort(key=lambda row: (row[0], row[3], row[1]))
        return rows[: max(1, int(limit))]

    def _park_robots_for_last_setup_wave(self, placement_events: List[Tuple[int, int, int, int]]) -> None:
        if not placement_events:
            return
        robot_count = len(self.robots)
        last_events = sorted(placement_events, key=lambda row: (row[0], row[1], row[2], row[3]))[
            -max(1, int(robot_count)) :
        ]
        final_setup_tick = max(row[0] for row in placement_events)
        robot_ids = sorted({int(row[1]) for row in last_events})
        for rid in robot_ids:
            robot = self._robot_by_id(rid)
            self._reserve_robot_wait_until(robot=robot, target_t=final_setup_tick)

    def _reserve_robot_wait_until(self, *, robot: RobotState, target_t: int) -> None:
        target_t = int(target_t)
        if target_t <= int(robot.last_t):
            return
        for t in range(int(robot.last_t) + 1, target_t + 1):
            if not self.planner.can_occupy(robot, int(t), int(robot.x), int(robot.y)):
                raise RuntimeError(
                    f"Cannot reserve wait for robot={robot.id} at t={t} cell=({robot.x},{robot.y})."
                )
            self.planner.reserve_footprint(robot, int(t), int(robot.x), int(robot.y))
        robot.last_t = int(target_t)

    def _verify_nest_layout(self, nest: NestState) -> None:
        failures: List[str] = []
        for _, sku, target_xy in nest.placed_jobs:
            actual = self.scheduler.pallets.get((int(target_xy[0]), int(target_xy[1])))
            if actual is None:
                failures.append(f"{target_xy}:missing expected={sku}")
                continue
            if int(actual) != int(sku):
                failures.append(f"{target_xy}:expected={sku} actual={actual}")
        if failures:
            preview = "; ".join(failures[:16])
            raise RuntimeError(f"Nest {nest.nest_id} integrity failed: {preview}")

    def _plan_order_phase(self, nests: List[NestState]) -> None:
        ordered_order_indexes = sorted(
            range(len(self.orders)),
            key=lambda idx: (
                int(sum(self.orders[idx].items.values())),
                int(len(self.orders[idx].items)),
                int(idx),
            ),
        )
        if not ordered_order_indexes:
            return

        dispatch_heap: List[Tuple[int, int, int, int]] = []
        rank = 0
        for nest in nests:
            start_xy = (int(nest.anchor[0]), 3)
            robots = sorted(
                (self._robot_by_id(rid) for rid in nest.robot_ids),
                key=lambda r: (
                    abs(int(r.x) - int(start_xy[0])) + abs(int(r.y) - int(start_xy[1])),
                    int(r.id),
                ),
            )
            for robot in robots:
                heapq.heappush(
                    dispatch_heap,
                    (int(robot.last_t), int(rank), int(robot.id), int(nest.nest_id)),
                )
                rank += 1

        completed: set[int] = set()
        self._completed_order_indices = completed
        for order_idx in ordered_order_indexes:
            self._check_global_limits_or_raise(
                collections.deque([idx for idx in ordered_order_indexes if idx not in completed])
            )
            self._enforce_unassigned_idle_limit(remaining_work=len(ordered_order_indexes) - len(completed))
            last_t, dispatch_rank, robot_id, nest_id = heapq.heappop(dispatch_heap)
            del last_t, dispatch_rank
            robot = self._robot_by_id(robot_id)
            nest = nests[nest_id]
            assigned_tick = max(0, int(robot.last_t) + 1)
            self.orders[order_idx].assigned_tick = assigned_tick
            fulfill_tick = self._plan_fixed_loop_order_job(
                robot=robot,
                order_idx=int(order_idx),
                order_counter=collections.Counter(self.orders[order_idx].items),
                nest_x=int(nest.anchor[0]),
                wait_limit=max(1, int(getattr(self.config, "conveyor_wait_stall_limit", 24))),
            )
            self.orders[order_idx].fulfilled_tick = int(fulfill_tick)
            completed.add(int(order_idx))
            self._maybe_log_progress(
                completed=len(completed),
                total_orders=len(ordered_order_indexes),
                dispatch_count=len(completed),
            )
            heapq.heappush(dispatch_heap, (int(robot.last_t), int(rank), int(robot.id), int(nest.nest_id)))
            rank += 1

    def _plan_fixed_loop_order_job(
        self,
        *,
        robot: RobotState,
        order_idx: int,
        order_counter: collections.Counter,
        nest_x: int,
        wait_limit: int,
    ) -> int:
        temp = self._clone_robot_state(robot)
        pending_actions: List[Tuple[int, int, str, int, int]] = []
        pending_paths: List[Tuple[RobotState, List[Tuple[int, int, int]]]] = []
        pending_footprints: List[Tuple[RobotState, int, int, int]] = []
        stall_cell: Tuple[int, int] | None = None
        stall_ticks = 0
        fulfill_tick = -1

        start_xy = (int(nest_x), 3)
        if getattr(robot, "last_job_kind", None) != "order":
            path = self._safe_plan_path_conveyor(temp, int(start_xy[0]), int(start_xy[1]))
            if not path and (int(temp.x), int(temp.y)) != start_xy:
                raise RuntimeError(
                    f"Robot {robot.id} could not reach order start {start_xy} for order {order_idx}."
                )
            if path:
                pending_paths.append((self._clone_robot_state(temp), path))
            pending_actions.extend(self._apply_moves_to_robot(temp, path))
        elif (int(temp.x), int(temp.y)) != start_xy:
            raise RuntimeError(
                f"Order robot {robot.id} expected at {start_xy}, found ({temp.x},{temp.y})."
            )

        remaining = collections.Counter(order_counter)

        def reserve_step(nx: int, ny: int) -> None:
            nonlocal stall_cell, stall_ticks
            while True:
                next_t = int(temp.last_t) + 1
                if self.planner.can_occupy(temp, next_t, int(nx), int(ny)):
                    step = [(next_t, int(nx), int(ny))]
                    pending_paths.append((self._clone_robot_state(temp), step))
                    if int(nx) != int(temp.x) or int(ny) != int(temp.y):
                        pending_actions.append((next_t, int(temp.id), "move", int(nx), int(ny)))
                        stall_cell = None
                        stall_ticks = 0
                    else:
                        here = (int(nx), int(ny))
                        if stall_cell == here:
                            stall_ticks += 1
                        else:
                            stall_cell = here
                            stall_ticks = 1
                        if stall_ticks >= int(wait_limit):
                            raise RuntimeError(
                                f"Robot {temp.id} stalled at tick {next_t} cell={here} waiting for path."
                            )
                    temp.last_t = int(next_t)
                    temp.x = int(nx)
                    temp.y = int(ny)
                    return

                if not self.planner.can_occupy(temp, next_t, int(temp.x), int(temp.y)):
                    raise RuntimeError(
                        f"Robot {temp.id} cannot wait safely at tick {next_t} cell=({temp.x},{temp.y})."
                    )
                wait_step = [(next_t, int(temp.x), int(temp.y))]
                pending_paths.append((self._clone_robot_state(temp), wait_step))
                here = (int(temp.x), int(temp.y))
                if stall_cell == here:
                    stall_ticks += 1
                else:
                    stall_cell = here
                    stall_ticks = 1
                if stall_ticks >= int(wait_limit):
                    raise RuntimeError(f"Robot {temp.id} stalled at tick {next_t} cell={here} waiting for robot ahead.")
                temp.last_t = int(next_t)

        def reserve_action(action: str, x: int, y: int) -> int:
            nonlocal stall_cell, stall_ticks
            while True:
                next_t = int(temp.last_t) + 1
                if self.planner.can_occupy(temp, next_t, int(temp.x), int(temp.y)):
                    if action == "pick" and not self._is_pick_target_static_at_time((int(x), int(y)), next_t):
                        reserve_step(int(temp.x), int(temp.y))
                        continue
                    pending_actions.append((next_t, int(temp.id), action, int(x), int(y)))
                    pending_footprints.append((self._clone_robot_state(temp), next_t, int(temp.x), int(temp.y)))
                    temp.last_t = int(next_t)
                    stall_cell = None
                    stall_ticks = 0
                    return next_t
                reserve_step(int(temp.x), int(temp.y))

        def pick_if_needed_from_north() -> None:
            px, py = int(temp.x), int(temp.y) - 1
            if not (0 <= px < self.state.width and 0 <= py < self.state.height):
                return
            sku = self.scheduler.pallets.get((px, py))
            if sku is None:
                return
            sku_i = int(sku)
            need = int(remaining.get(sku_i, 0))
            while need > 0:
                reserve_action("pick", int(px), int(py))
                temp.storage[sku_i] += 1
                remaining[sku_i] -= 1
                if int(remaining[sku_i]) <= 0:
                    remaining.pop(sku_i, None)
                need = int(remaining.get(sku_i, 0))

        pick_if_needed_from_north()
        for _ in range(11):
            reserve_step(int(temp.x) + 1, int(temp.y))
            pick_if_needed_from_north()
        for _ in range(2):
            reserve_step(int(temp.x), int(temp.y) - 1)
        if (int(temp.x), int(temp.y)) != (int(nest_x) + 11, 1):
            raise RuntimeError(f"Robot {temp.id} failed east-leg checkpoint for order {order_idx}.")

        pick_if_needed_from_north()
        for _ in range(12):
            reserve_step(int(temp.x) - 1, int(temp.y))
            pick_if_needed_from_north()
        if (int(temp.x), int(temp.y)) != (int(nest_x) - 1, 1):
            raise RuntimeError(f"Robot {temp.id} failed west-leg checkpoint for order {order_idx}.")

        missing = self._format_missing_requirements(required=collections.Counter(order_counter), have=temp.storage)
        if missing != "none":
            raise RuntimeError(f"Order {order_idx} missing inventory before fulfill: {missing}")

        reserve_step(int(temp.x), int(temp.y) - 1)
        fulfill_tick = reserve_action("fulfill", int(temp.x), int(temp.y))
        temp.storage.clear()

        reserve_step(int(temp.x) - 1, int(temp.y))
        for _ in range(3):
            reserve_step(int(temp.x), int(temp.y) + 1)
        for _ in range(2):
            reserve_step(int(temp.x) + 1, int(temp.y))
        if (int(temp.x), int(temp.y)) != start_xy:
            raise RuntimeError(
                f"Robot {temp.id} failed loop return checkpoint; expected {start_xy}, got ({temp.x},{temp.y})."
            )

        if not self._can_commit_pending_actions(pending_actions):
            raise RuntimeError(f"Order plan candidate rejected by validator for robot={robot.id} order={order_idx}.")

        self._commit_plan(
            robot=robot,
            temp_robot=temp,
            pending_actions=pending_actions,
            pending_paths=pending_paths,
            pending_footprints=pending_footprints,
        )
        setattr(robot, "last_job_kind", "order")
        return int(fulfill_tick)

    def _enforce_unassigned_idle_limit(self, *, remaining_work: int) -> None:
        if remaining_work <= 0:
            return
        current = max((int(r.last_t) for r in self.robots), default=-1)
        limit = max(1, int(getattr(self.config, "max_idle_ticks", 24)))
        for robot in self.robots:
            idle = int(current) - int(robot.last_t)
            if idle > limit:
                raise RuntimeError(
                    f"Robot {robot.id} idle too long: idle_ticks={idle} limit={limit} current_tick={current}."
                )

    def _robot_by_id(self, robot_id: int) -> RobotState:
        rid = int(robot_id)
        for robot in self.robots:
            if int(robot.id) == rid:
                return robot
        raise KeyError(f"Robot not found: {robot_id}")

    def _maybe_log_progress(self, *, completed: int, total_orders: int, dispatch_count: int) -> None:
        order_interval = max(1, int(getattr(self.config, "progress_order_interval", getattr(self.config, "progress_every", 100))))
        tick_interval = max(1, int(getattr(self.config, "progress_tick_interval", 1000)))
        current_makespan = max((int(r.last_t) for r in self.robots), default=-1)
        elapsed_s = self._elapsed_plan_seconds()
        tick_hit = current_makespan >= self._next_tick_progress
        order_hit = completed % order_interval == 0
        if not tick_hit and not order_hit:
            return
        if tick_hit:
            while self._next_tick_progress <= current_makespan:
                self._next_tick_progress += tick_interval
        now_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[solver] {now_stamp} tick={current_makespan} fulfilled={completed}/{total_orders} "
            f"dispatches={dispatch_count} runtime={elapsed_s:.1f}s"
        )
        self._log(
            f"progress ts={now_stamp} tick={current_makespan} fulfilled={completed}/{total_orders} "
            f"dispatches={dispatch_count} runtime_s={elapsed_s:.3f}"
        )
