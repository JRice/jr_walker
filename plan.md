# Warehouse Robot Scheduler — AI Implementation Spec

Source problem: 

## 0. Core Principles

* Optimize **makespan (ticks)** only.
* Use **greedy incremental planning**:

  * Plan one robot at a time.
  * Once planned, paths are immutable (reservation table constraint).
* Use **spacetime A*** for all non-track movement.
* Use **deterministic track execution** (no A*) for order fulfillment.
* Enforce strict **collision constraints** across `(t, x, y)`.

---

## 1. Configuration System

### Files

* `config/config.toml` → all tunable parameters
* `config/run_number.txt` → persistent run ID

### Required Config Fields

```
[system]
max_ticks = 16000
max_runtime_minutes = 10
idle_tick_limit = 24
log_interval_ticks = 1000
log_interval_orders = 100

[pathfinding]
max_expansions = <int>   # safety cap for A*
strict_mode = false

[orders]
stride = 1

[nests]
x_positions = [15, 35]

[warehouse]
width = 40
height = 60
```

### Run ID Logic

```
if run_number.txt missing or empty:
    write "1"
else:
    read N → write N+1
```

---

## 2. Data Model

### 2.1 Spacetime Warehouse (Reservation Table)

**Structure:**

```
Warehouse:
    occupancy: Dict[(t, x, y) → EntityID]
```

* Each cell contains **at most one entity** (robot OR pallet).
* Time is **explicit dimension**.
* Unchanged entities are implicitly projected forward unless moved.

---

### 2.2 Entities

#### Robot

```
Robot:
    id: int
    position[t]: (x, y)
    inventory: Dict[sku → int]
    job: Optional[Job]
    nest_id: int
    available_at_tick: int
```

Constraints:

* Carries items for **one order only**
* Picks cost **1 tick (no movement)**

---

#### Pallet

```
Pallet:
    id: int
    sku: int
    position[t]: (x, y)
    reserved_for_nest: Optional[nest_id]
```

---

#### Docking Model

* Dock = adjacency in cardinal direction
* Robot may dock to **multiple pallets (N/E/S/W)**
* Movement = **rigid translation**
* Docked pallets occupy independent cells and must obey collision rules

---

### 2.3 Orders

```
Order:
    id: int
    items: Dict[sku → quantity]
    total_items: int
    assigned_tick: Optional[int]
    fulfilled_tick: Optional[int]
```

* Orders sorted by `total_items ASC`
* Apply `stride` during load

---

### 2.4 Jobs

```
JobType:
    SETUP
    ORDER
    WAIT
    DONE
```

Each job has a planner.

---

## 3. Core Algorithms

---

### 3.1 Spacetime A* (Reservation-Based)

**State:**

```
(t, x, y, docked_offsets)
```

**Constraints:**

* No two entities share `(t,x,y)`
* If `strict_mode=True`, forbid edge swaps:

  ```
  (A at t,x1,y1 → x2,y2) AND (B at t,x2,y2 → x1,y1)
  ```
* Docked pallets move with robot → composite occupancy check

**Cost:**

```
g = ticks
h = Manhattan distance
f = g + h
```

**Termination:**

* Stop if `max_expansions` exceeded → raise exception

---

### 3.2 Nearest Queries

All use A* distance (not Manhattan):

* `nearest_pallet(sku, not_reserved)`
* `nearest_robot(position, available_at_tick)`
* `nearest_empty_3x3x3(region)`

---

### 3.3 3×3×3 Empty Region

A region is valid if:

```
∀ dx ∈ [-1,0,1]
∀ dy ∈ [-1,0,1]
∀ dt ∈ [0,1,2]:
    (t+dt, x+dx, y+dy) is empty
```

---

## 4. Phase 1: Nest Construction

---

### 4.1 Nest Layout

For each `nest_x`:

```
Row 1: y=0, x ∈ [nest_x ... nest_x+9]
Row 2: y=2, x ∈ [nest_x ... nest_x+9]
```

Total: 20 pallets (SKUs 1–20)

---

### 4.2 Robot Assignment

At tick 0:

```
for each nest:
    assign nearest available robots
    if none assigned → error
```

Robots remain permanently assigned to that nest.

---

### 4.3 Scheduling Strategy

Global:

```
planned_pallets = set()
```

Per nest:

```
unplanned_skus = {1..20}
available_positions = stack of 20 coordinates
```

Loop:

```
while unplanned_skus not empty:
    pallet = nearest pallet with SKU ∈ unplanned_skus and not in planned_pallets
    robot = nearest available robot to pallet
    dest = available_positions.pop()

    assign SETUP job immediately
    mark pallet as planned
```

---

### 4.4 Setup Job Planner

Steps:

1. Move robot to adjacent cell of pallet
2. Dock
3. If docked from non-north:

   * Move to nearest 3×3×3 empty region
   * Undock
   * Move to south of pallet
   * Re-dock
4. Use **dock-aware A*** to move pallet to destination
5. Undock

Robot becomes available next tick.

---

### 4.5 Synchronization

* Final `N` pallets (N = robots in nest):

  * Robots **wait until last pallet placed**
* Each nest transitions independently to next phase

---

## 5. Phase 2: Order Fulfillment

---

### 5.1 Initialization

Per nest:

* Sort robots by distance to `[nest_x, 3]`
* Assign orders to robots by:

  ```
  earliest available robot first
  shortest order first
  ```

---

### 5.2 Order Job — Deterministic Track

No A*. Only sequential planning.

---

#### Track Definition

Start at `[nest_x, 3]`

**Segment 1:**

```
Move east 11
At each step:
    if pallet north AND needed SKU:
        pick (1 tick)
    else:
        move
```

**Segment 2:**

```
Move north 2
Assert position == [nest_x+11, 1]
```

**Segment 3:**

```
Move west 12
Pick as needed (same rule)
```

**Segment 4:**

```
Assert position == [nest_x-1, 1]
```

**Segment 5:**

```
Verify inventory complete → else error
Move north 1
fulfill()
Move west 1
Move south 3
Move east 2
Assert position == [nest_x, 3]
```

---

### 5.3 Movement Constraints

* Single-file flow
* If next cell occupied:

  * WAIT (counts as idle)
* If idle > `idle_tick_limit` → error

---

### 5.4 Picking

* 1 item per tick
* No movement during pick

---

### 5.5 Execution Loop

```
while orders remain:
    select robot with lowest available_at_tick
    assign next shortest order
    plan job
```

---

## 6. Scheduling Constraints

* No robot idle > `idle_tick_limit` (except after all orders complete)
* Abort if:

  * `tick > max_ticks`
  * runtime exceeds limit
  * any invariant violation

---

## 7. Output

---

### 7.1 Logging

Log every:

* `log_interval_ticks`
* OR `log_interval_orders`

Format:

```
[tick] [timestamp] [fulfilled_orders]
```

---

### 7.2 Output File

Path:

```
output/run{run_id}_{ticks}tick_{fulfilled_orders}order_solution.txt
```

Write on:

* normal completion
* any failure

Only log **state changes** (moves, picks, dock/undock).

---

### 7.3 PNG Graph

If all orders fulfilled:

```
media/run{run_id}_{ticks}tick_{fulfilled_orders}order_rate.png
```

Plot:

* X: ticks
* Y: fulfilled orders

---

### 7.4 Validation

Run `validator.py` after output (unless early exception prevents it).

---

## 8. Testing Targets

Focus on:

* A* correctness (with and without strict mode)
* Docked movement collision handling
* Reservation table integrity
* Nest construction validity
* Track execution correctness
* Idle timeout enforcement

---

## 9. Architecture (Recommended)

---

### Modules

```
core/
    warehouse.py        # reservation table
    entities.py         # Robot, Pallet, Order
    config.py

planning/
    astar.py            # spacetime A*
    dock_astar.py       # dock-aware A*
    heuristics.py

jobs/
    setup_job.py
    order_job.py

scheduler/
    scheduler.py        # main loop

utils/
    io.py
    logging.py
    metrics.py
```

---

### Key Pattern

* **Reservation Table Pattern** → central constraint system
* **Greedy Sequential Planning** → no backtracking
* **Strategy Pattern (Jobs)** → each job encapsulates planning logic

---

## 10. Invariants (Must Never Be Violated)

* One entity per `(t,x,y)`
* No illegal swaps in strict mode
* Docked entities move rigidly
* Orders fully satisfied before fulfill()
* Robots remain within assigned nest domain during order phase
