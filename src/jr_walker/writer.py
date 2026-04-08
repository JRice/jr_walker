from pathlib import Path
from typing import Iterable, Tuple


Action = Tuple[int, int, str, int, int]


def write_actions(actions: Iterable[Action], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for t, robot_id, action, x, y in actions:
            handle.write(f"{t} {robot_id} {action} {x} {y}\n")

    return output_path
