from typing import Iterable, List, Tuple

import numpy as np

from jr_walker.entities import Robot
from jr_walker.routing import find_path, reserve_path
from jr_walker.sim import RobotState


def adjacent_cells(width: int, height: int, x: int, y: int) -> List[Tuple[int, int]]:
    candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    out = []
    for cx, cy in candidates:
        if 0 <= cx < width and 0 <= cy < height:
            out.append((cx, cy))
    return out


def nearest_perimeter_cell(width: int, height: int, x: int, y: int) -> Tuple[int, int]:
    options = [(x, 0), (x, height - 1), (0, y), (width - 1, y)]
    return min(options, key=lambda p: abs(p[0] - x) + abs(p[1] - y))


class ReservationPlanner:
    def __init__(self, static_blocked: np.ndarray, width: int, height: int, max_time: int = 50000):
        self.width = width
        self.height = height
        self.max_time = max_time

        # 0 = free, 2 = static pallet, 3 = reserved robot occupancy
        self.reservation_table = np.zeros((max_time, height, width), dtype=np.int8)
        static_layer = static_blocked.astype(np.int8) * 2
        self.reservation_table[:] = static_layer

    def _to_robot(self, robot_state: RobotState) -> Robot:
        robot = Robot(robot_state.id, robot_state.x, robot_state.y)
        robot.docks = dict(robot_state.docks)
        return robot

    def plan_path(self, robot_state: RobotState, target_x: int, target_y: int) -> List[Tuple[int, int, int]]:
        robot = self._to_robot(robot_state)
        return find_path(
            robot,
            robot_state.last_t,
            robot_state.x,
            robot_state.y,
            target_x,
            target_y,
            self.reservation_table,
        )

    def reserve_path(self, robot_state: RobotState, path: Iterable[Tuple[int, int, int]]) -> None:
        robot = self._to_robot(robot_state)
        reserve_path(robot, list(path), self.reservation_table)

    def can_occupy(self, robot_state: RobotState, timestep: int, x: int, y: int) -> bool:
        if not (0 <= timestep < self.max_time):
            return False

        robot = self._to_robot(robot_state)
        for fx, fy in robot.get_footprint(x, y):
            if not (0 <= fx < self.width and 0 <= fy < self.height):
                return False
            # > 1 means static pallet or reserved dynamic occupancy
            if self.reservation_table[timestep, fy, fx] > 1:
                return False
        return True

    def reserve_footprint(self, robot_state: RobotState, timestep: int, x: int, y: int) -> None:
        if not (0 <= timestep < self.max_time):
            raise ValueError(f"Timestep {timestep} is outside reservation horizon {self.max_time}")

        if not self.can_occupy(robot_state, timestep, x, y):
            raise ValueError(
                f"Cannot reserve footprint for robot={robot_state.id} at t={timestep}, x={x}, y={y}"
            )

        robot = self._to_robot(robot_state)
        for fx, fy in robot.get_footprint(x, y):
            if 0 <= fx < self.width and 0 <= fy < self.height:
                self.reservation_table[timestep, fy, fx] = 3
