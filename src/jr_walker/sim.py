import collections
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


Action = Tuple[int, int, str, int, int]


@dataclass
class RobotState:
    id: int
    x: int
    y: int
    last_t: int = -1
    storage: collections.Counter = field(default_factory=collections.Counter)
    docks: Dict[Tuple[int, int], int] = field(default_factory=dict)


class ActionLog:
    def __init__(self) -> None:
        self._actions: List[Action] = []
        self._seen = set()

    def add(self, timestep: int, robot_id: int, action: str, x: int, y: int) -> None:
        key = (timestep, robot_id)
        if key in self._seen:
            raise ValueError(f"Duplicate action for timestep={timestep}, robot={robot_id}")
        self._seen.add(key)
        self._actions.append((timestep, robot_id, action, x, y))

    def sorted_actions(self) -> List[Action]:
        return sorted(self._actions, key=lambda a: (a[0], a[1]))

    def has_action(self, timestep: int, robot_id: int) -> bool:
        return (timestep, robot_id) in self._seen
