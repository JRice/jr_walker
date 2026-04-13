import collections
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


GridCell = Tuple[int, int]


@dataclass(frozen=True)
class MacroAction:
    """Task-planner macro action that ignores low-level grid dynamics."""

    name: str
    target: GridCell | None = None
    params: Dict[str, int] = field(default_factory=dict)


class SetupTaskPlanner:
    """
    Task Planner layer for setup relocation.
    Produces coarse macro-actions; motion details are delegated to Motion Planner.
    """

    def build_setup_relocation_macros(
        self,
        *,
        stand_xy: GridCell,
        source_xy: GridCell,
        target_xy: GridCell,
        requires_local_maneuver: bool,
    ) -> List[MacroAction]:
        macros: List[MacroAction] = [
            MacroAction(name="move_to_stand", target=stand_xy),
            MacroAction(name="dock_source", target=source_xy),
        ]
        if requires_local_maneuver:
            macros.append(
                MacroAction(
                    name="maneuver_pivot",
                    target=source_xy,
                    params={"target_x": int(target_xy[0]), "target_y": int(target_xy[1])},
                )
            )
        macros.extend(
            [
                MacroAction(name="carry_to_target", target=target_xy),
                MacroAction(name="undock_target", target=target_xy),
            ]
        )
        return macros


@dataclass(frozen=True)
class PrimitiveStep:
    action: str
    x: int
    y: int


@dataclass(frozen=True)
class ManeuverPlan:
    steps: Tuple[PrimitiveStep, ...]
    expanded_states: int


class MiniBoxMotionPlanner:
    """
    Motion Planner layer for local manipulation.
    Solves micro-maneuvers in a small local bounding box using BFS.
    """

    def __init__(self, width: int, height: int, box_radius: int = 2):
        self.width = int(width)
        self.height = int(height)
        self.box_radius = max(1, int(box_radius))

    def plan_pivot(
        self,
        *,
        robot_xy: GridCell,
        pallet_xy: GridCell,
        start_offset: GridCell,
        target_offset: GridCell,
        static_blocked_cells: Iterable[GridCell],
        box_radius: int | None = None,
    ) -> ManeuverPlan | None:
        """
        Maneuver A: Pivot / re-grab.
        Converts a docked pallet orientation from `start_offset` to `target_offset` by:
        1) undock
        2) local robot reposition around static pallet
        3) dock again from target side
        """
        rx, ry = robot_xy
        px, py = pallet_xy
        expected_pallet_xy = (rx + start_offset[0], ry + start_offset[1])
        if expected_pallet_xy != pallet_xy:
            return None

        desired_robot_xy = (px - target_offset[0], py - target_offset[1])
        if abs(desired_robot_xy[0] - px) + abs(desired_robot_xy[1] - py) != 1:
            return None

        walk = self.plan_local_walk(
            start_xy=robot_xy,
            goal_xy=desired_robot_xy,
            pallet_xy=pallet_xy,
            static_blocked_cells=static_blocked_cells,
            box_radius=box_radius,
        )
        if walk is None:
            return None

        steps: List[PrimitiveStep] = [PrimitiveStep("undock", px, py)]
        for wx, wy in walk:
            steps.append(PrimitiveStep("move", wx, wy))
        steps.append(PrimitiveStep("dock", px, py))
        return ManeuverPlan(steps=tuple(steps), expanded_states=len(walk))

    def plan_local_walk(
        self,
        *,
        start_xy: GridCell,
        goal_xy: GridCell,
        pallet_xy: GridCell,
        static_blocked_cells: Iterable[GridCell],
        box_radius: int | None = None,
    ) -> List[GridCell] | None:
        radius = self.box_radius if box_radius is None else max(1, int(box_radius))
        min_x = max(0, int(start_xy[0]) - radius)
        max_x = min(self.width - 1, int(start_xy[0]) + radius)
        min_y = max(0, int(start_xy[1]) - radius)
        max_y = min(self.height - 1, int(start_xy[1]) + radius)

        if not (min_x <= goal_xy[0] <= max_x and min_y <= goal_xy[1] <= max_y):
            return None
        if not (min_x <= pallet_xy[0] <= max_x and min_y <= pallet_xy[1] <= max_y):
            return None

        blocked: set[GridCell] = set(static_blocked_cells)
        blocked.discard((int(pallet_xy[0]), int(pallet_xy[1])))
        blocked.add((int(pallet_xy[0]), int(pallet_xy[1])))

        start = (int(start_xy[0]), int(start_xy[1]))
        goal = (int(goal_xy[0]), int(goal_xy[1]))
        if start == goal:
            return []
        if start in blocked or goal in blocked:
            return None

        queue: collections.deque[GridCell] = collections.deque([start])
        prev: Dict[GridCell, GridCell | None] = {start: None}

        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == goal:
                break
            for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                if not (min_x <= nx <= max_x and min_y <= ny <= max_y):
                    continue
                if (nx, ny) in blocked:
                    continue
                if (nx, ny) in prev:
                    continue
                prev[(nx, ny)] = (cx, cy)
                queue.append((nx, ny))

        if goal not in prev:
            return None

        path: List[GridCell] = []
        node: GridCell | None = goal
        while node is not None and node != start:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path
