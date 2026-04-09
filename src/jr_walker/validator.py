from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


GRID_WIDTH = 60
GRID_HEIGHT = 40
ALLOWED_ACTIONS = {"move", "pick", "fulfill", "dock", "undock"}


class ValidationError(ValueError):
    """Raised when a worklist or submission fails validation."""


@dataclass(slots=True)
class Action:
    timestep: int
    robot_id: int
    action_type: str
    x: int
    y: int
    line_no: int
    raw_line: str


@dataclass(slots=True)
class RobotState:
    robot_id: int
    x: int
    y: int
    storage: Counter[int] = field(default_factory=Counter)
    docked_pallets: list[int] = field(default_factory=list)


@dataclass(slots=True)
class PalletState:
    pallet_id: int
    x: int
    y: int
    sku: int
    docked_to_robot: int | None = None


@dataclass(slots=True)
class OrderState:
    order_id: int
    items: Counter[int]
    fulfilled: bool = False


@dataclass(slots=True)
class ValidatorState:
    next_timestep: int
    fulfilled_orders: int
    total_orders: int
    robots: list[RobotState]
    pallets: list[PalletState]
    orders: list[OrderState]


def _key(x: int, y: int) -> int:
    return 100 * y + x


def _in_bounds(x: int, y: int) -> bool:
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT


def _is_adjacent(ax: int, ay: int, bx: int, by: int) -> bool:
    dx = abs(ax - bx)
    dy = abs(ay - by)
    return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _non_comment_non_blank_lines(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("\ufeff")
        if stripped and not stripped.startswith("#"):
            out.append(stripped)
    return out


def parse_worklist_text(text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], list[Counter[int]]]:
    lines = _non_comment_non_blank_lines(text)
    if not lines:
        raise ValidationError("error WL18: worklist file is empty")

    idx = 0
    used_positions: set[tuple[int, int]] = set()

    robot_count = _parse_int(lines[idx])
    if robot_count is None or robot_count < 0:
        raise ValidationError(
            f'error WL1: invalid robot count "{lines[idx]}" - must be a non-negative integer'
        )
    idx += 1

    robot_positions: list[tuple[int, int]] = []
    for robot_id in range(robot_count):
        if idx >= len(lines):
            raise ValidationError(
                f"error WL2: expected {robot_count} robot positions but file ended after {robot_id} robots"
            )
        raw = lines[idx]
        idx += 1
        parts = raw.split()
        if len(parts) < 2:
            raise ValidationError(
                f'error WL3: invalid robot position format on line {idx} "{raw}" - expected "x y"'
            )
        x = _parse_int(parts[0])
        y = _parse_int(parts[1])
        if x is None or y is None:
            raise ValidationError(
                f'error WL4: invalid robot coordinates on line {idx} "{raw}" - x and y must be integers'
            )
        if not _in_bounds(x, y):
            raise ValidationError(
                f"error WL5: robot {robot_id} position ({x}, {y}) is out of bounds (grid is 60x40)"
            )
        if (x, y) in used_positions:
            raise ValidationError(f"error WL6: robot {robot_id} at ({x}, {y}) overlaps with another robot")
        used_positions.add((x, y))
        robot_positions.append((x, y))

    if idx >= len(lines):
        raise ValidationError("error WL7: missing pallet count - file ended after robot positions")

    pallet_count = _parse_int(lines[idx])
    if pallet_count is None or pallet_count < 0:
        raise ValidationError(
            f'error WL7: invalid pallet count "{lines[idx]}" - must be a non-negative integer'
        )
    idx += 1

    pallets: list[tuple[int, int, int]] = []
    for pallet_id in range(pallet_count):
        if idx >= len(lines):
            raise ValidationError(f"error WL8: expected {pallet_count} pallets but file ended after {pallet_id} pallets")
        raw = lines[idx]
        idx += 1
        parts = raw.split()
        if len(parts) < 3:
            raise ValidationError(
                f'error WL9: invalid pallet data format on line {idx} "{raw}" - expected "x y sku"'
            )
        x = _parse_int(parts[0])
        y = _parse_int(parts[1])
        sku = _parse_int(parts[2])
        if x is None or y is None or sku is None:
            raise ValidationError(
                f'error WL10: invalid pallet data on line {idx} "{raw}" - x, y, and sku must be integers'
            )
        if not _in_bounds(x, y):
            raise ValidationError(
                f"error WL11: pallet {pallet_id} position ({x}, {y}) is out of bounds (grid is 60x40)"
            )
        if (x, y) in used_positions:
            for rid, (rx, ry) in enumerate(robot_positions):
                if rx == x and ry == y:
                    raise ValidationError(f"error WL13: pallet {pallet_id} at ({x}, {y}) overlaps with robot {rid}")
            raise ValidationError(f"error WL12: pallet {pallet_id} at ({x}, {y}) overlaps with another pallet")
        used_positions.add((x, y))
        pallets.append((x, y, sku))

    if idx >= len(lines):
        raise ValidationError("error WL14: missing order count - file ended after pallet data")

    order_count = _parse_int(lines[idx])
    if order_count is None or order_count < 0:
        raise ValidationError(
            f'error WL14: invalid order count "{lines[idx]}" - must be a non-negative integer'
        )
    idx += 1

    orders: list[Counter[int]] = []
    for order_id in range(order_count):
        if idx >= len(lines):
            raise ValidationError(f"error WL15: expected {order_count} orders but file ended after {order_id} orders")
        raw = lines[idx]
        idx += 1
        sku_strs = [s for s in raw.split() if s]
        if not sku_strs:
            raise ValidationError(f"error WL17: order {order_id} on line {idx} is empty (no SKUs)")
        bag: Counter[int] = Counter()
        for sku_text in sku_strs:
            sku = _parse_int(sku_text)
            if sku is None or sku < 0:
                raise ValidationError(
                    f'error WL16: invalid SKU "{sku_text}" in order {order_id} on line {idx} - SKUs must be non-negative integers'
                )
            bag[sku] += 1
        orders.append(bag)

    return robot_positions, pallets, orders


def parse_worklist_file(path: str | Path) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], list[Counter[int]]]:
    return parse_worklist_text(Path(path).read_text(encoding="utf-8"))


class SubmissionValidator:
    """
    Validates submission actions with browser-parity rules.

    Supports both one-shot file validation and incremental line-by-line validation.
    """

    def __init__(self, *, worklist_path: str | Path | None = None, worklist_text: str | None = None) -> None:
        if (worklist_path is None) == (worklist_text is None):
            raise ValueError("Provide exactly one of worklist_path or worklist_text.")

        if worklist_text is not None:
            robot_positions, pallets, orders = parse_worklist_text(worklist_text)
        else:
            robot_positions, pallets, orders = parse_worklist_file(worklist_path)  # type: ignore[arg-type]

        self.robots: list[RobotState] = [
            RobotState(robot_id=rid, x=x, y=y) for rid, (x, y) in enumerate(robot_positions)
        ]
        self.pallets: list[PalletState] = [
            PalletState(pallet_id=pid, x=x, y=y, sku=sku) for pid, (x, y, sku) in enumerate(pallets)
        ]
        self.orders: list[OrderState] = [
            OrderState(order_id=oid, items=Counter(items)) for oid, items in enumerate(orders)
        ]
        self.total_orders = len(self.orders)
        self.fulfilled_orders = 0

        self._pallet_by_pos: dict[int, PalletState] = {}
        for pallet in self.pallets:
            key = _key(pallet.x, pallet.y)
            if key in self._pallet_by_pos:
                prev = self._pallet_by_pos[key]
                raise ValidationError(
                    f"error ST2: multiple pallets at position ({pallet.x}, {pallet.y}) - pallet {prev.pallet_id} and pallet {pallet.pallet_id}"
                )
            self._pallet_by_pos[key] = pallet

        self._robot_by_pos: dict[tuple[int, int], RobotState] = {}
        for robot in self.robots:
            pos = (robot.x, robot.y)
            if pos in self._robot_by_pos:
                prev = self._robot_by_pos[pos]
                raise ValidationError(
                    f"error ST1: multiple robots at position ({robot.x}, {robot.y}) - robot {prev.robot_id} and robot {robot.robot_id}"
                )
            if _key(robot.x, robot.y) in self._pallet_by_pos:
                raise ValidationError(
                    f"error ST3: pallet {self._pallet_by_pos[_key(robot.x, robot.y)].pallet_id} and robot {robot.robot_id} both at position ({robot.x}, {robot.y})"
                )
            self._robot_by_pos[pos] = robot

        self._next_timestep = 0
        self._current_timestep: int | None = None
        self._pending_actions: list[Action] = []
        self._robots_seen_this_timestep: set[int] = set()
        self._last_seen_timestep = -1
        self._line_count = 0
        self.completed = self.fulfilled_orders == self.total_orders

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the validator state."""
        return {
            "next_timestep": self._next_timestep,
            "current_timestep": self._current_timestep,
            "fulfilled_orders": self.fulfilled_orders,
            "total_orders": self.total_orders,
            "last_seen_timestep": self._last_seen_timestep,
            "line_count": self._line_count,
            "robots_seen_this_timestep": sorted(self._robots_seen_this_timestep),
            "pending_actions": [
                {
                    "timestep": a.timestep,
                    "robot_id": a.robot_id,
                    "action_type": a.action_type,
                    "x": a.x,
                    "y": a.y,
                    "line_no": a.line_no,
                    "raw_line": a.raw_line,
                }
                for a in self._pending_actions
            ],
            "robots": [
                {
                    "robot_id": r.robot_id,
                    "x": r.x,
                    "y": r.y,
                    "storage": dict(r.storage),
                    "docked_pallets": list(r.docked_pallets),
                }
                for r in self.robots
            ],
            "pallets": [
                {
                    "pallet_id": p.pallet_id,
                    "x": p.x,
                    "y": p.y,
                    "sku": p.sku,
                    "docked_to_robot": p.docked_to_robot,
                }
                for p in self.pallets
            ],
            "orders": [
                {"order_id": o.order_id, "items": dict(o.items), "fulfilled": o.fulfilled} for o in self.orders
            ],
            "completed": self.completed,
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> SubmissionValidator:
        """Restore from `snapshot()` output."""
        obj = cls.__new__(cls)
        obj.total_orders = int(snapshot["total_orders"])
        obj.fulfilled_orders = int(snapshot["fulfilled_orders"])
        obj._next_timestep = int(snapshot["next_timestep"])
        current = snapshot.get("current_timestep")
        obj._current_timestep = None if current is None else int(current)
        obj._last_seen_timestep = int(snapshot["last_seen_timestep"])
        obj._line_count = int(snapshot["line_count"])
        obj._robots_seen_this_timestep = set(int(rid) for rid in snapshot["robots_seen_this_timestep"])
        obj.completed = bool(snapshot["completed"])

        obj.robots = [
            RobotState(
                robot_id=int(r["robot_id"]),
                x=int(r["x"]),
                y=int(r["y"]),
                storage=Counter({int(k): int(v) for k, v in dict(r["storage"]).items()}),
                docked_pallets=[int(pid) for pid in r["docked_pallets"]],
            )
            for r in snapshot["robots"]
        ]
        obj.pallets = [
            PalletState(
                pallet_id=int(p["pallet_id"]),
                x=int(p["x"]),
                y=int(p["y"]),
                sku=int(p["sku"]),
                docked_to_robot=None if p["docked_to_robot"] is None else int(p["docked_to_robot"]),
            )
            for p in snapshot["pallets"]
        ]
        obj.orders = [
            OrderState(
                order_id=int(o["order_id"]),
                items=Counter({int(k): int(v) for k, v in dict(o["items"]).items()}),
                fulfilled=bool(o["fulfilled"]),
            )
            for o in snapshot["orders"]
        ]
        obj._pending_actions = [
            Action(
                timestep=int(a["timestep"]),
                robot_id=int(a["robot_id"]),
                action_type=str(a["action_type"]),
                x=int(a["x"]),
                y=int(a["y"]),
                line_no=int(a["line_no"]),
                raw_line=str(a["raw_line"]),
            )
            for a in snapshot["pending_actions"]
        ]

        obj._pallet_by_pos = {}
        for pallet in obj.pallets:
            obj._pallet_by_pos[_key(pallet.x, pallet.y)] = pallet
        obj._robot_by_pos = {(r.x, r.y): r for r in obj.robots}
        return obj

    def get_state(self) -> ValidatorState:
        return ValidatorState(
            next_timestep=self._next_timestep,
            fulfilled_orders=self.fulfilled_orders,
            total_orders=self.total_orders,
            robots=self.robots,
            pallets=self.pallets,
            orders=self.orders,
        )

    def validate_line(self, line: str) -> None:
        """
        Parse and validate one action line.

        Raises ValidationError immediately on the first fault.
        """
        self._line_count += 1
        stripped = line.strip().lstrip("\ufeff")
        if not stripped or stripped.startswith("#"):
            return

        action = self._parse_action_line(stripped, line_no=self._line_count)
        if self.completed:
            return

        if self._current_timestep is None:
            self._current_timestep = action.timestep
            self._robots_seen_this_timestep.clear()
        elif action.timestep > self._current_timestep:
            self._flush_up_to(action.timestep)
            self._current_timestep = action.timestep
            self._robots_seen_this_timestep.clear()

        if action.robot_id in self._robots_seen_this_timestep:
            raise ValidationError(
                f"error SB5: robot {action.robot_id} has multiple actions at timestep {action.timestep} (line {action.line_no})"
            )
        self._robots_seen_this_timestep.add(action.robot_id)
        self._pending_actions.append(action)

    def finalize(self) -> ValidatorState:
        """Flush pending lines and validate the remaining timesteps."""
        if self.completed:
            return self.get_state()
        if self._current_timestep is None:
            return self.get_state()
        self._flush_up_to(self._current_timestep + 1)
        return self.get_state()

    def validate_file(self, path: str | Path) -> ValidatorState:
        text = Path(path).read_text(encoding="utf-8")
        for line in text.splitlines():
            self.validate_line(line)
        return self.finalize()

    def _parse_action_line(self, raw: str, *, line_no: int) -> Action:
        parts = raw.split()
        if len(parts) < 5:
            raise ValidationError(
                f'error SB1: invalid action format on line {line_no} "{raw}" - expected "<timestep> <robot_id> <action> <x> <y>"'
            )

        timestep = _parse_int(parts[0])
        if timestep is None or timestep < 0:
            raise ValidationError(
                f'error SB2: invalid timestep "{parts[0]}" on line {line_no} - must be a non-negative integer'
            )
        if timestep < self._last_seen_timestep:
            raise ValidationError(
                f"error SB3: actions not in order on line {line_no} - timestep {timestep} comes after timestep {self._last_seen_timestep}"
            )
        self._last_seen_timestep = timestep

        robot_id = _parse_int(parts[1])
        if robot_id is None or robot_id < 0:
            raise ValidationError(
                f'error SB4: invalid robot ID "{parts[1]}" on line {line_no} - must be a non-negative integer'
            )

        action_type = parts[2].lower()
        if action_type not in ALLOWED_ACTIONS:
            allowed = ", ".join(sorted(ALLOWED_ACTIONS))
            raise ValidationError(
                f'error SB6: invalid action type "{parts[2]}" on line {line_no} - must be one of: {allowed}'
            )

        x = _parse_int(parts[3])
        y = _parse_int(parts[4])
        if x is None or y is None:
            raise ValidationError(
                f"error SB7: invalid coordinates ({parts[3]}, {parts[4]}) on line {line_no} - must be integers"
            )
        return Action(
            timestep=timestep,
            robot_id=robot_id,
            action_type=action_type,
            x=x,
            y=y,
            line_no=line_no,
            raw_line=raw,
        )

    def _flush_up_to(self, new_timestep: int) -> None:
        while self._next_timestep < new_timestep and not self.completed:
            if self._current_timestep == self._next_timestep:
                actions = self._pending_actions
                self._pending_actions = []
            else:
                actions = []
            self._run_tick(self._next_timestep, actions)
            self._next_timestep += 1

    def _robot_at(self, x: int, y: int) -> RobotState | None:
        return self._robot_by_pos.get((x, y))

    def _pallet_at(self, x: int, y: int) -> PalletState | None:
        return self._pallet_by_pos.get(_key(x, y))

    def _run_tick(self, timestep: int, actions: list[Action]) -> None:
        if not actions:
            return

        moving_actions = [a for a in actions if a.action_type == "move"]
        moving_robot_ids = {a.robot_id for a in moving_actions}
        moving_docked_keys: set[int] = set()
        for move in moving_actions:
            robot = self._get_robot_or_error(move.robot_id)
            for pallet_id in robot.docked_pallets:
                pallet = self.pallets[pallet_id]
                moving_docked_keys.add(_key(pallet.x, pallet.y))

        for action in actions:
            try:
                self._validate_action_legality(action, moving_robot_ids, moving_docked_keys)
            except ValidationError as exc:
                raise ValidationError(f"timestep {timestep}: {exc}") from None

        move_conflict = self._validate_move_conflicts(moving_actions, moving_robot_ids)
        if move_conflict is not None:
            raise ValidationError(f"timestep {timestep}: {move_conflict}")

        dock_conflict = self._validate_simultaneous_dock_conflicts(actions)
        if dock_conflict is not None:
            raise ValidationError(f"timestep {timestep}: {dock_conflict}")

        for move in moving_actions:
            moved_dock_conflict = self._validate_moved_docked_pallet_collisions(
                move, moving_robot_ids, moving_docked_keys
            )
            if moved_dock_conflict is not None:
                raise ValidationError(f"timestep {timestep}: {moved_dock_conflict}")

        for action in actions:
            if action.action_type == "undock":
                self._apply_undock(action)
        for action in actions:
            if action.action_type == "pick":
                self._apply_pick(action)
        for action in actions:
            if action.action_type == "dock":
                self._apply_dock(action)
        for action in actions:
            if action.action_type == "move":
                self._apply_move(action)
        self._robot_by_pos = {(r.x, r.y): r for r in self.robots}
        for action in actions:
            if action.action_type == "fulfill":
                self._apply_fulfill(action)

        self.completed = self.fulfilled_orders == self.total_orders

    def _get_robot_or_error(self, robot_id: int) -> RobotState:
        if robot_id < 0 or robot_id >= len(self.robots):
            raise ValidationError(f"error A: robot {robot_id} does not exist")
        return self.robots[robot_id]

    def _validate_action_legality(
        self, action: Action, moving_robot_ids: set[int], moving_docked_keys: set[int]
    ) -> None:
        robot = self._get_robot_or_error(action.robot_id)

        if action.action_type == "move":
            if not _is_adjacent(robot.x, robot.y, action.x, action.y):
                raise ValidationError(f"error B: robot {robot.robot_id} cannot move to non-adjacent cell")
            if not _in_bounds(action.x, action.y):
                raise ValidationError(f"error C: robot {robot.robot_id} cannot move out of bounds")
            pallet = self._pallet_at(action.x, action.y)
            if pallet is not None and pallet.docked_to_robot != robot.robot_id:
                target_key = _key(action.x, action.y)
                if target_key not in moving_docked_keys:
                    who = (
                        f"pallet {pallet.pallet_id} docked to robot {pallet.docked_to_robot}"
                        if pallet.docked_to_robot is not None
                        else f"undocked pallet {pallet.pallet_id}"
                    )
                    raise ValidationError(
                        f"error E: robot {robot.robot_id} cannot move to ({action.x}, {action.y}) occupied by {who}"
                    )
            return

        if action.action_type == "pick":
            if not _is_adjacent(robot.x, robot.y, action.x, action.y):
                raise ValidationError(f"error I: robot {robot.robot_id} cannot pick from non-adjacent cell")
            if self._pallet_at(action.x, action.y) is None:
                raise ValidationError("error K: no pallet at pick target")
            return

        if action.action_type == "fulfill":
            if not (robot.x == 0 or robot.x == GRID_WIDTH - 1 or robot.y == 0 or robot.y == GRID_HEIGHT - 1):
                raise ValidationError(
                    f"error L: robot {robot.robot_id} not in fulfillment zone (must be on perimeter)"
                )
            if not robot.storage:
                raise ValidationError(f"error AB: robot {robot.robot_id} has empty storage")
            for order in self.orders:
                if not order.fulfilled and order.items == robot.storage:
                    return
            raise ValidationError("error M: robot storage doesn't match any order")

        if action.action_type == "dock":
            if not _is_adjacent(robot.x, robot.y, action.x, action.y):
                raise ValidationError(f"error N: robot {robot.robot_id} cannot dock non-adjacent pallet")
            pallet = self._pallet_at(action.x, action.y)
            if pallet is None:
                raise ValidationError("error P: no pallet at dock target")
            if pallet.docked_to_robot is not None:
                raise ValidationError("error Q: pallet already docked")
            if len(robot.docked_pallets) >= 4:
                raise ValidationError("error S: robot already has 4 docked pallets")
            return

        if action.action_type == "undock":
            pallet = self._pallet_at(action.x, action.y)
            if pallet is None:
                raise ValidationError("error U: no pallet at undock target")
            if pallet.docked_to_robot != robot.robot_id:
                raise ValidationError("error V: pallet not docked to this robot")
            return

        raise ValidationError(f"Unknown action type: {action.action_type}")

    def _validate_move_conflicts(self, moving_actions: list[Action], moving_robot_ids: set[int]) -> str | None:
        if not moving_actions:
            return None

        target_to_robot: dict[int, int] = {}
        for move in moving_actions:
            target_key = _key(move.x, move.y)
            other = target_to_robot.get(target_key)
            if other is not None:
                return f"error Y: robots {other} and {move.robot_id} both moving to same cell"
            target_to_robot[target_key] = move.robot_id

        for move in moving_actions:
            robot = self._robot_at(move.x, move.y)
            if robot is not None and robot.robot_id not in moving_robot_ids:
                return (
                    f"error Z: robot {move.robot_id} moving to ({move.x}, {move.y}) "
                    f"occupied by stationary robot {robot.robot_id}"
                )
        return None

    def _validate_simultaneous_dock_conflicts(self, actions: list[Action]) -> str | None:
        dock_actions = [a for a in actions if a.action_type == "dock"]
        if len(dock_actions) <= 1:
            return None
        target_to_robot: dict[int, int] = {}
        for dock in dock_actions:
            target_key = _key(dock.x, dock.y)
            other = target_to_robot.get(target_key)
            if other is not None:
                return (
                    f"error X: robots {other} and {dock.robot_id} both trying to dock pallet "
                    f"at ({dock.x}, {dock.y}) simultaneously"
                )
            target_to_robot[target_key] = dock.robot_id
        return None

    def _validate_moved_docked_pallet_collisions(
        self, move: Action, moving_robot_ids: set[int], moving_docked_keys: set[int]
    ) -> str | None:
        robot = self.robots[move.robot_id]
        if not robot.docked_pallets:
            return None

        dx = move.x - robot.x
        dy = move.y - robot.y
        for pallet_id in robot.docked_pallets:
            pallet = self.pallets[pallet_id]
            nx = pallet.x + dx
            ny = pallet.y + dy

            if not _in_bounds(nx, ny):
                return (
                    f"error F: robot {robot.robot_id}'s docked pallet {pallet_id} "
                    f"would move out of bounds to ({nx}, {ny})"
                )

            for other_robot in self.robots:
                if other_robot.robot_id == robot.robot_id:
                    continue
                if other_robot.x == nx and other_robot.y == ny:
                    if other_robot.robot_id in moving_robot_ids:
                        continue
                    return (
                        f"error G: robot {robot.robot_id}'s docked pallet {pallet_id} "
                        f"would collide with robot {other_robot.robot_id} at ({nx}, {ny})"
                    )

            other_pallet = self._pallet_at(nx, ny)
            if other_pallet is not None and other_pallet.docked_to_robot != robot.robot_id:
                target_key = _key(nx, ny)
                if target_key in moving_docked_keys:
                    continue
                who = (
                    f"pallet {other_pallet.pallet_id} docked to robot {other_pallet.docked_to_robot}"
                    if other_pallet.docked_to_robot is not None
                    else f"undocked pallet {other_pallet.pallet_id}"
                )
                return (
                    f"error H: robot {robot.robot_id}'s docked pallet {pallet_id} "
                    f"would collide with {who} at ({nx}, {ny})"
                )
        return None

    def _apply_undock(self, action: Action) -> None:
        robot = self.robots[action.robot_id]
        pallet = self._pallet_at(action.x, action.y)
        if pallet is None:
            return
        pallet.docked_to_robot = None
        robot.docked_pallets = [pid for pid in robot.docked_pallets if pid != pallet.pallet_id]

    def _apply_pick(self, action: Action) -> None:
        robot = self.robots[action.robot_id]
        pallet = self._pallet_at(action.x, action.y)
        if pallet is None:
            return
        robot.storage[pallet.sku] += 1

    def _apply_dock(self, action: Action) -> None:
        robot = self.robots[action.robot_id]
        pallet = self._pallet_at(action.x, action.y)
        if pallet is None:
            return
        pallet.docked_to_robot = robot.robot_id
        robot.docked_pallets.append(pallet.pallet_id)

    def _apply_move(self, action: Action) -> None:
        robot = self.robots[action.robot_id]
        dx = action.x - robot.x
        dy = action.y - robot.y

        robot.x = action.x
        robot.y = action.y

        for pallet_id in robot.docked_pallets:
            pallet = self.pallets[pallet_id]
            del self._pallet_by_pos[_key(pallet.x, pallet.y)]
            pallet.x += dx
            pallet.y += dy
            self._pallet_by_pos[_key(pallet.x, pallet.y)] = pallet

    def _apply_fulfill(self, action: Action) -> None:
        robot = self.robots[action.robot_id]
        for order in self.orders:
            if order.fulfilled:
                continue
            if order.items == robot.storage:
                order.fulfilled = True
                robot.storage.clear()
                self.fulfilled_orders += 1
                return


def validate_solution_file(
    solution_path: str | Path, *, worklist_path: str | Path = "docs/BIG_ORDER.txt"
) -> ValidatorState:
    """
    Convenience one-shot validator.

    Raises ValidationError on the first problem encountered.
    """
    validator = SubmissionValidator(worklist_path=worklist_path)
    return validator.validate_file(solution_path)
