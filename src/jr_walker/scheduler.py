import collections
from typing import Dict, Iterable, List, Tuple

from jr_walker.planner import adjacent_cells, nearest_perimeter_cell
from jr_walker.sim import RobotState


def manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


class GreedyScheduler:
    def __init__(self, width: int, height: int, pallets: Dict[Tuple[int, int], int]):
        self.width = width
        self.height = height
        self.pallets: Dict[Tuple[int, int], int] = dict(pallets)
        self.sku_to_pallets: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
        self.pick_cells_by_pallet: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self.sku_to_pallets = collections.defaultdict(list)
        self.pick_cells_by_pallet = {}

        for (px, py), sku in self.pallets.items():
            self.sku_to_pallets[sku].append((px, py))
            valid_cells = []
            for cx, cy in adjacent_cells(self.width, self.height, px, py):
                # Pick cells must not be occupied by static pallets.
                if (cx, cy) in self.pallets:
                    continue
                valid_cells.append((cx, cy))
            self.pick_cells_by_pallet[(px, py)] = valid_cells

    def has_sku(self, sku: int) -> bool:
        return len(self.sku_to_pallets.get(sku, [])) > 0

    def pallet_cells_for_sku(self, sku: int) -> List[Tuple[int, int]]:
        return list(self.sku_to_pallets.get(sku, []))

    def pick_cells_for_pallet(self, pallet_xy: Tuple[int, int]) -> List[Tuple[int, int]]:
        return list(self.pick_cells_by_pallet.get(pallet_xy, []))

    def move_pallet(self, from_xy: Tuple[int, int], to_xy: Tuple[int, int]) -> None:
        if from_xy not in self.pallets:
            raise ValueError(f"Cannot move pallet from {from_xy}: missing source.")
        if to_xy in self.pallets:
            raise ValueError(f"Cannot move pallet to {to_xy}: destination occupied.")
        sku = self.pallets.pop(from_xy)
        self.pallets[to_xy] = sku
        self._rebuild_indexes()

    def rank_robots_for_order(self, order: collections.Counter, robots: Iterable[RobotState]) -> List[int]:
        unique_skus = list(order.keys())
        item_count = sum(order.values())

        scores = []
        for robot in robots:
            nearest_pick = self._nearest_pick_distance((robot.x, robot.y), unique_skus)
            fulfill_dist = self._nearest_perimeter_distance(robot.x, robot.y)
            available_at = robot.last_t + 1
            score = available_at + nearest_pick + item_count + fulfill_dist
            scores.append((score, robot.id))

        scores.sort()
        return [robot_id for _, robot_id in scores]

    def candidate_pick_options(
        self, remaining: collections.Counter, current_xy: Tuple[int, int]
    ) -> List[Tuple[int, int, Tuple[int, int], Tuple[int, int]]]:
        """
        Returns sorted candidate options:
        (estimated_distance, sku, pallet_xy, pick_cell_xy)
        """
        options = []
        for sku, qty in remaining.items():
            if qty <= 0:
                continue
            for pallet_xy in self.sku_to_pallets.get(sku, []):
                for pick_cell_xy in self.pick_cells_by_pallet.get(pallet_xy, []):
                    dist = manhattan(current_xy, pick_cell_xy)
                    options.append((dist, sku, pallet_xy, pick_cell_xy))

        options.sort(key=lambda o: o[0])
        return options

    def best_fulfill_cell(self, x: int, y: int) -> Tuple[int, int]:
        return nearest_perimeter_cell(self.width, self.height, x, y)

    def _nearest_pick_distance(self, current_xy: Tuple[int, int], skus: Iterable[int]) -> int:
        best = float("inf")
        for sku in skus:
            for pallet_xy in self.sku_to_pallets.get(sku, []):
                for pick_cell_xy in self.pick_cells_by_pallet.get(pallet_xy, []):
                    best = min(best, manhattan(current_xy, pick_cell_xy))
        if best == float("inf"):
            return 10**9
        return int(best)

    def _nearest_perimeter_distance(self, x: int, y: int) -> int:
        return min(x, self.width - 1 - x, y, self.height - 1 - y)
