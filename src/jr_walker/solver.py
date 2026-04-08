import collections
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from jr_walker.planner import ReservationPlanner
from jr_walker.scheduler import GreedyScheduler
from jr_walker.sim import ActionLog, RobotState
from jr_walker.writer import write_actions


@dataclass
class SolverConfig:
    max_time: int = 50000
    progress_every: int = 50
    output_path: Path = Path("output/solution.txt")


class WarehouseSolver:
    def __init__(self, warehouse_state, config: SolverConfig | None = None):
        self.state = warehouse_state
        self.config = config or SolverConfig()

        self.robots: List[RobotState] = [
            RobotState(id=rid, x=x, y=y) for rid, (x, y) in enumerate(self.state.robots)
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
            static_grid=self.state.grid,
        )

    def solve(self) -> Tuple[Path, List[Tuple[int, int, str, int, int]]]:
        for order_idx, order in enumerate(self.state.orders):
            ok = self._plan_order(order_idx, order)
            if not ok:
                raise RuntimeError(f"Could not build a feasible plan for order {order_idx}")

            if (order_idx + 1) % self.config.progress_every == 0:
                current_makespan = max(robot.last_t for robot in self.robots)
                print(
                    f"[solver] planned {order_idx + 1}/{len(self.state.orders)} orders, "
                    f"current makespan={current_makespan}"
                )

        sorted_actions = self.actions.sorted_actions()
        output_path = write_actions(sorted_actions, self.config.output_path)
        return output_path, sorted_actions

    def _plan_order(self, order_idx: int, order: collections.Counter) -> bool:
        ranked_robot_ids = self.scheduler.rank_robots_for_order(order, self.robots)
        for robot_id in ranked_robot_ids:
            robot = self.robots[robot_id]
            ok = self._plan_order_for_robot(order_idx, order, robot)
            if ok:
                return True
        return False

    def _plan_order_for_robot(
        self, order_idx: int, order: collections.Counter, robot: RobotState
    ) -> bool:
        remaining = collections.Counter(order)

        while sum(remaining.values()) > 0:
            options = self.scheduler.candidate_pick_options(remaining, (robot.x, robot.y))
            selected = None
            for _, sku, pallet_xy, pick_cell_xy in options:
                target_x, target_y = pick_cell_xy
                path = self.planner.plan_path(robot, target_x, target_y)

                if path or (robot.x == target_x and robot.y == target_y):
                    selected = (sku, pallet_xy, pick_cell_xy, path)
                    break

            if selected is None:
                return False

            sku, pallet_xy, pick_cell_xy, path = selected
            self._commit_moves(robot, path)

            pick_t = robot.last_t + 1
            pallet_x, pallet_y = pallet_xy
            self.actions.add(pick_t, robot.id, "pick", pallet_x, pallet_y)
            self.planner.reserve_footprint(robot, pick_t, robot.x, robot.y)
            robot.last_t = pick_t
            robot.storage[sku] += 1

            remaining[sku] -= 1
            if remaining[sku] <= 0:
                del remaining[sku]

        fulfill_x, fulfill_y = self.scheduler.best_fulfill_cell(robot.x, robot.y)
        fulfill_path = self.planner.plan_path(robot, fulfill_x, fulfill_y)
        if not fulfill_path and (robot.x != fulfill_x or robot.y != fulfill_y):
            return False

        self._commit_moves(robot, fulfill_path)
        fulfill_t = robot.last_t + 1
        self.actions.add(fulfill_t, robot.id, "fulfill", fulfill_x, fulfill_y)
        self.planner.reserve_footprint(robot, fulfill_t, robot.x, robot.y)
        robot.last_t = fulfill_t
        robot.storage.clear()

        return True

    def _commit_moves(self, robot: RobotState, path: List[Tuple[int, int, int]]) -> None:
        if not path:
            return

        prev_x, prev_y = robot.x, robot.y
        for t, x, y in path:
            # Space-time A* can emit WAIT steps (same x/y). In the submission format,
            # waiting is represented by omitting an action for that robot/timestep.
            if x != prev_x or y != prev_y:
                self.actions.add(t, robot.id, "move", x, y)
            prev_x, prev_y = x, y

        self.planner.reserve_path(robot, path)
        robot.last_t = path[-1][0]
        robot.x = path[-1][1]
        robot.y = path[-1][2]
