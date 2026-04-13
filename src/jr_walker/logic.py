import collections
from typing import Dict, Iterable, List, Tuple
from abc import ABC, abstractmethod

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


class Suggestion(ABC):
    @property
    @abstractmethod
    def expected_cost(self) -> float:
        ...

    @property
    @abstractmethod
    def expected_gain(self) -> float:
        ...

    @property
    @abstractmethod
    def center(self) -> Tuple[int, int]:
        ...

    def score(self) -> float:
        return self.expected_gain - self.expected_cost


class RelocateSuggestion(Suggestion):
    def __init__(self, job, scheduler, remaining_job_factor_fn=None):
        self.job = job
        self.scheduler = scheduler
        self.remaining_job_factor_fn = remaining_job_factor_fn
        self._cost = -1.0
        self._gain = -1.0
        self._center = (-1, -1)

    @property
    def expected_cost(self) -> float:
        if self._cost < 0:
            source_pallets = self.scheduler.pallet_cells_for_sku(self.job.sku)
            if not source_pallets or self.job.preferred_target_xy is None:
                self._cost = float("inf")
            else:
                closest_source = min(source_pallets, key=lambda p: manhattan_distance(p, self.job.preferred_target_xy))
                self._cost = float(manhattan_distance(closest_source, self.job.preferred_target_xy))
        return self._cost

    @property
    def expected_gain(self) -> float:
        if self._gain < 0:
            self._gain = self.job.score
        return self._gain

    def scale_gain(self, factor: float):
        self._gain = self.expected_gain * factor

    def remaining_job_factor(self) -> float:
        if self.remaining_job_factor_fn is None:
            return 1.0
        return max(0.0, float(self.remaining_job_factor_fn(self.job.sku)))

    def score(self) -> float:
        # Scale relocation value by how many remaining orders still need this SKU.
        return (self.expected_gain * self.remaining_job_factor()) - self.expected_cost

    @property
    def center(self) -> Tuple[int, int]:
        if self._center == (-1, -1):
            source_pallets = self.scheduler.pallet_cells_for_sku(self.job.sku)
            if not source_pallets:
                self._center = self.job.hotspot
            else:
                self._center = min(source_pallets, key=lambda p: manhattan_distance(p, self.job.hotspot))
        return self._center

    def __str__(self) -> str:
        source_xy = self.center
        dest_xy = self.job.preferred_target_xy
        if dest_xy is None:
            dest_xy = (
                int(self.job.hotspot[0] + self.job.placement_offset[0]),
                int(self.job.hotspot[1] + self.job.placement_offset[1]),
            )
        return (
            "RelocateSuggestion: "
            f"source ({source_xy[0]},{source_xy[1]}) to dest ({dest_xy[0]},{dest_xy[1]})"
        )

class DockSuggestion(Suggestion):
    def __init__(self, sku: int, plan: List[int], gain: float, pallet_xy: Tuple[int, int]):
        self.sku = sku
        self.plan = plan
        self._gain = gain
        self._center = pallet_xy

    @property
    def expected_cost(self) -> float:
        # The cost is to travel to the pallet and dock it. For now, we treat this as
        # a high-priority strategic move, so the cost is negligible.
        return 0.0

    @property
    def expected_gain(self) -> float:
        return self._gain

    @property
    def center(self) -> Tuple[int, int]:
        return self._center

    def __str__(self) -> str:
        order_stream = " ".join(str(int(order_id)) for order_id in self.plan)
        return (
            "DockSuggestion: "
            f"robot <R> plans {len(self.plan)} orders with SKU {self.sku}: ({order_stream})"
        )


class SetupSuggestion(Suggestion):
    def __init__(self, job):
        self.job = job
        self._cost = 0.0
        self._gain = 10000.0

    @property
    def expected_cost(self) -> float:
        return self._cost

    @property
    def expected_gain(self) -> float:
        return self._gain

    @property
    def center(self) -> Tuple[int, int]:
        return self.job.source_xy

    def __str__(self) -> str:
        assigned_robot_id = getattr(self, "assigned_robot_id", None)
        robot_label = "?" if assigned_robot_id is None else str(int(assigned_robot_id))
        sx, sy = self.job.source_xy
        tx, ty = self.job.target_xy
        return (
            "SetupSuggestion: "
            f"source ({sx},{sy}) to dest ({tx},{ty}) for robot {robot_label}"
        )


class OrderSuggestion(Suggestion):
    def __init__(self, order_idx, order, cluster, order_gain_constant, warehouse_width, warehouse_height, scheduler):
        self.order_idx = order_idx
        self.order = order
        self.cluster = cluster
        self.preferred_pallet_cells_by_sku: Dict[int, List[Tuple[int, int]]] = {}
        for sku in order.keys():
            sku_i = int(sku)
            preferred: List[Tuple[int, int]] = []
            cluster_xy = cluster.get(sku_i)
            if cluster_xy is not None:
                preferred.append((int(cluster_xy[0]), int(cluster_xy[1])))
            self.preferred_pallet_cells_by_sku[sku_i] = preferred
        self._order_gain_constant = order_gain_constant
        self._width = warehouse_width
        self._height = warehouse_height
        self._scheduler = scheduler
        self._cost = -1.0
        self._center = (-1, -1)

    def _nearest_available_edge_distance(self, current_xy: Tuple[int, int]) -> float:
        cx, cy = current_xy
        perimeter_cells = set()
        for x in range(self._width):
            perimeter_cells.add((x, 0))
            perimeter_cells.add((x, self._height - 1))
        for y in range(self._height):
            perimeter_cells.add((0, y))
            perimeter_cells.add((self._width - 1, y))

        best = float("inf")
        for ex, ey in perimeter_cells:
            if (ex, ey) in self._scheduler.pallets:
                continue
            best = min(best, manhattan_distance((cx, cy), (ex, ey)))
        return best

    @property
    def expected_cost(self) -> float:
        if self._cost < 0:
            xs = [pos[0] for pos in self.cluster.values()]
            ys = [pos[1] for pos in self.cluster.values()]
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            center_xy = self.center
            dist_to_edge = self._nearest_available_edge_distance(center_xy)
            self._cost = float(span_x * span_y + dist_to_edge)
        return self._cost

    @property
    def expected_gain(self) -> float:
        return self._order_gain_constant

    @property
    def center(self) -> Tuple[int, int]:
        if self._center == (-1, -1):
            xs = [pos[0] for pos in self.cluster.values()]
            ys = [pos[1] for pos in self.cluster.values()]
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            self._center = (int(min(xs) + span_x / 2), int(min(ys) + span_y / 2))
        return self._center

    def __str__(self) -> str:
        sku_parts = " ".join(
            f"{int(qty)}x{int(sku)}" for sku, qty in sorted(self.order.items(), key=lambda row: int(row[0]))
        )
        return f"OrderSuggestion: order ({sku_parts})"


class EdgeAwareOrderScorer:
    """
    Estimates order effort using marginal detour to a fulfill edge:
    picks that sit near the current route to edge are scored as low cost.
    """

    def __init__(
        self,
        scheduler,
        width: int,
        height: int,
        hot_spots: List[Tuple[int, int]] | None = None,
    ):
        self.scheduler = scheduler
        self.width = width
        self.height = height
        self.hot_spots = [p for p in (hot_spots or []) if self._in_bounds(*p)]

    def rank_orders_for_robot(
        self,
        robot,
        order_ids: Iterable[int],
        orders: List[collections.Counter],
        top_k: int | None = None,
    ) -> List[int]:
        scored = []
        for order_idx in order_ids:
            score = self.estimate_order_cost_for_robot(robot, orders[order_idx])
            scored.append((score, order_idx))

        scored.sort(key=lambda t: (t[0], t[1]))
        if top_k is not None:
            scored = scored[:top_k]
        return [order_idx for _, order_idx in scored]

    def estimate_order_cost_for_robot(self, robot, order: collections.Counter) -> float:
        start_xy = (robot.x, robot.y)
        edge_candidates = self._edge_candidates(start_xy)
        best = float("inf")
        for edge_xy in edge_candidates:
            est = self._estimate_with_edge(start_xy, edge_xy, order)
            if est < best:
                best = est
        if best == float("inf"):
            return best
        return (robot.last_t + 1) + best

    def estimate_order_cost(self, order: collections.Counter) -> float:
        """
        Intrinsic order cost estimate for dispatch strategy:
        - Includes travel between picks
        - Includes all pick actions
        - Includes travel from last pick to nearest available edge
        - Includes fulfill action
        - Excludes robot travel to the first pick
        """
        if not order:
            return 1.0

        best_total = float("inf")
        for sku in order.keys():
            first_pick = self._best_pick_distance_for_sku((0, 0), sku, ignore_current=True)
            if first_pick is None:
                continue
            _, first_xy = first_pick
            total = self._estimate_from_first_pick(order, sku, first_xy)
            if total < best_total:
                best_total = total
        return best_total

    def _estimate_from_first_pick(
        self,
        order: collections.Counter,
        first_sku: int,
        first_pick_xy: Tuple[int, int],
    ) -> float:
        if first_sku not in order:
            return float("inf")

        remaining = set(order.keys())
        total = float(order[first_sku])  # pick count at first stand cell
        current_xy = first_pick_xy
        remaining.remove(first_sku)

        while remaining:
            choice = None
            for sku in remaining:
                best_pick = self._best_pick_distance_for_sku(current_xy, sku)
                if best_pick is None:
                    continue
                travel, pick_xy = best_pick
                qty = order[sku]
                candidate = (travel, -qty, sku, pick_xy)
                if choice is None or candidate < choice:
                    choice = candidate

            if choice is None:
                return float("inf")

            travel, neg_qty, sku, pick_xy = choice
            qty = -neg_qty
            total += float(travel + qty)
            current_xy = pick_xy
            remaining.remove(sku)

        to_edge = self._nearest_available_edge_distance(current_xy)
        if to_edge == float("inf"):
            return float("inf")
        total += float(to_edge)
        total += 1.0  # fulfill action
        return total

    def _best_pick_distance_for_sku(
        self,
        current_xy: Tuple[int, int],
        sku: int,
        ignore_current: bool = False,
    ) -> Tuple[int, Tuple[int, int]] | None:
        if not self.scheduler.has_sku(sku):
            return None
        best = None
        for pallet_xy in self.scheduler.pallet_cells_for_sku(sku):
            for pick_xy in self.scheduler.pick_cells_for_pallet(pallet_xy):
                if ignore_current:
                    travel = 0
                else:
                    travel = manhattan_distance(current_xy, pick_xy)
                candidate = (travel, pick_xy)
                if best is None or candidate < best:
                    best = candidate
        return best

    def _nearest_available_edge_distance(self, current_xy: Tuple[int, int]) -> float:
        cx, cy = current_xy
        perimeter_cells = set()
        for x in range(self.width):
            perimeter_cells.add((x, 0))
            perimeter_cells.add((x, self.height - 1))
        for y in range(self.height):
            perimeter_cells.add((0, y))
            perimeter_cells.add((self.width - 1, y))

        best = float("inf")
        for ex, ey in perimeter_cells:
            if (ex, ey) in self.scheduler.pallets:
                continue
            best = min(best, manhattan_distance((cx, cy), (ex, ey)))
        return best

    def _estimate_with_edge(
        self,
        start_xy: Tuple[int, int],
        edge_xy: Tuple[int, int],
        order: collections.Counter,
    ) -> float:
        remaining = set(order.keys())
        current_xy = start_xy
        total = 0.0

        while remaining:
            choice = None
            for sku in remaining:
                qty = order[sku]
                best_pick = self._best_pick_for_sku(current_xy, edge_xy, sku)
                if best_pick is None:
                    continue
                detour, travel, pick_xy = best_pick
                candidate = (detour, travel, -qty, sku, pick_xy)
                if choice is None or candidate < choice:
                    choice = candidate

            if choice is None:
                return float("inf")

            _, travel, neg_qty, sku, pick_xy = choice
            qty = -neg_qty
            total += travel
            total += qty  # pick actions at the same stand cell
            current_xy = pick_xy
            remaining.remove(sku)

        total += manhattan_distance(current_xy, edge_xy)
        total += 1  # fulfill action
        return total

    def _best_pick_for_sku(
        self,
        current_xy: Tuple[int, int],
        edge_xy: Tuple[int, int],
        sku: int,
    ) -> Tuple[int, int, Tuple[int, int]] | None:
        if not self.scheduler.has_sku(sku):
            return None

        base_to_edge = manhattan_distance(current_xy, edge_xy)
        best = None
        for pallet_xy in self.scheduler.pallet_cells_for_sku(sku):
            for pick_xy in self.scheduler.pick_cells_for_pallet(pallet_xy):
                travel = manhattan_distance(current_xy, pick_xy)
                tail = manhattan_distance(pick_xy, edge_xy)
                detour = travel + tail - base_to_edge
                candidate = (detour, travel, pick_xy)
                if best is None or candidate < best:
                    best = candidate
        return best

    def _edge_candidates(self, start_xy: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = start_xy
        candidates = set(self.hot_spots)
        candidates.add((x, 0))
        candidates.add((x, self.height - 1))
        candidates.add((0, y))
        candidates.add((self.width - 1, y))
        return [p for p in candidates if self._in_bounds(*p)]

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

class OrderOptimizer:
    def __init__(self, pallets: Dict[Tuple[int, int], int]):
        """
        pallets: The {(x,y): sku} dictionary from our WarehouseState
        """
        # We need a reverse lookup: {sku: [(x1,y1), (x2,y2)...]}
        self.sku_locations = collections.defaultdict(list)
        for (x, y), sku in pallets.items():
            self.sku_locations[sku].append((x, y))

    def find_tightest_cluster(self, order_skus: List[int]) -> Tuple[Dict[int, Tuple[int, int]], int]:
        """
        Given a list of unique SKUs needed for an order, finds the tightest physical grouping 
        of pallets that satisfy the order.
        
        Returns:
            best_cluster: {sku: (target_x, target_y)} mapping
            bounding_box_score: The perimeter of the bounding box containing the cluster
        """
        # 1. Sort SKUs by rarity (fewest pallets in the warehouse first)
        # This prevents O(N^K) explosion by forcing the cluster to form around the most constrained variable.
        sorted_skus = sorted(order_skus, key=lambda s: len(self.sku_locations[s]))
        
        anchor_sku = sorted_skus[0]
        other_skus = sorted_skus[1:]
        
        best_cluster = None
        min_bounding_box = float('inf')
        
        # 2. Iterate through the pallets of the RAREST SKU
        for anchor_pos in self.sku_locations[anchor_sku]:
            current_cluster = {anchor_sku: anchor_pos}
            
            # 3. For every other SKU, greedily grab the closest pallet to the anchor
            for sku in other_skus:
                closest_pallet = min(self.sku_locations[sku], 
                                     key=lambda p: manhattan_distance(anchor_pos, p))
                current_cluster[sku] = closest_pallet
                
            # 4. Calculate the Bounding Box of this cluster
            xs = [pos[0] for pos in current_cluster.values()]
            ys = [pos[1] for pos in current_cluster.values()]
            
            # Perimeter of the bounding box (smaller = tighter cluster)
            bbox_perimeter = 2 * ((max(xs) - min(xs)) + (max(ys) - min(ys)))
            
            if bbox_perimeter < min_bounding_box:
                min_bounding_box = bbox_perimeter
                best_cluster = current_cluster
                
        return best_cluster, min_bounding_box

    def sort_orders_by_cluster_efficiency(self, orders: List[collections.Counter]):
        """
        Takes all 1000 orders and sorts them by how tightly clustered their required pallets are.
        Orders that can be picked in a 5x5 area will bubble to the top.
        """
        scored_orders = []
        for i, order in enumerate(orders):
            unique_skus = list(order.keys())
            _, bbox_score = self.find_tightest_cluster(unique_skus)
            scored_orders.append({
                "order_idx": i,
                "order": order,
                "cluster_score": bbox_score
            })
            
        # Sort by smallest bounding box
        scored_orders.sort(key=lambda x: x["cluster_score"])
        return scored_orders

    def analyze_big_order(pallets, orders):
        print(f"Initialized Optimizer with {len(pallets)} pallets and {len(orders)} orders...\n")
        
        # 2. Run the Optimization
        optimizer = OrderOptimizer(pallets)
        scored_orders = optimizer.sort_orders_by_cluster_efficiency(orders)
        
        # 3. Print the Results
        print("## TOP 5 EASIEST ORDERS (Tightest Clusters - Do these first!)")
        print("-" * 75)
        for i in range(5):
            o = scored_orders[i]
            score = o['cluster_score']
            print(f"Rank {i+1:03} | Order ID: {o['order_idx']:>4} | Perimeter Score: {score:>3} | SKUs: {dict(o['order'])}")
            
        print("\n## BOTTOM 5 HARDEST ORDERS (Widest Clusters - Dock candidates!)")
        print("-" * 75)
        for i in range(1, 6):
            o = scored_orders[-i]
            score = o['cluster_score']
            print(f"Rank {1000-i+1:03} | Order ID: {o['order_idx']:>4} | Perimeter Score: {score:>3} | SKUs: {dict(o['order'])}")

    def print_distribution_report(orders):
        """Takes the parsed orders list from WarehouseState and prints the SKU distribution."""
        
        all_skus = []
        for order in orders:
            # order is a collections.Counter, elements() unpacks it back to a flat list
            all_skus.extend(list(order.elements()))

        counter = collections.Counter(all_skus)
        total_picks = len(all_skus)
        
        print("-" * 30)
        print("DISTRIBUTION ANALYSIS")
        print(f"Total Orders: {len(orders)}")
        print(f"Total Individual Picks: {total_picks}")
        print(f"Unique SKUs: {len(counter)}")
        print("-" * 30)
        
        running_total = 0
        for i, (sku, count) in enumerate(counter.most_common(10), 1):
            percentage = (count / total_picks) * 100
            running_total += percentage
            print(f"{i}. SKU {sku: >3}: {count: >5} picks ({percentage:.1f}%)")
            
        print("-" * 30)
