# jr_walker: Program Notes & Architecture Cheat Sheet

## Domain Overview
* **Goal:** Fulfill 1,000 orders using 5 robots on a 60x40 grid in the minimum number of timesteps.
* **Key Rules:** Robots pick SKUs from pallets into storage, can dock pallets to carry them, and fulfill orders at the perimeter. No collisions (1 entity per cell).

## Core Files & Responsibilities

### Entry & Orchestration
* `main.py`: CLI entry point. Parses args, loads hyperparameters from TOML (`config.toml`), and runs the `WarehouseSolver`. Triggers post-optimization (LNS) and metadata persistence.
* `view.py` (`WarehouseState`): Parses the `BIG_ORDER.txt` input (grid, pallets, robots, orders) and handles visualization.

### Solving & Routing
* `solver.py` (`WarehouseSolver`): The core orchestrator. 
  * Builds a queue of `Suggestion` objects (`OrderSuggestion`, `RelocateSuggestion`), sorts them by `gain - cost`, and executes them greedily with the next available robot.
  * Runs **Large Neighborhood Search (LNS)** as a post-optimization pipeline step to shift actions backwards in time to compress the makespan.
* `routing.py`: Low-level pathfinding.
  * Uses **Space-Time A*** (`find_path`) with a 3D reservation tensor `(time, y, x)` to avoid collisions.
* `entities.py` (`Robot`): Maintains dynamic state for pathfinding, especially calculating 3D footprints when pallets are docked.
* `logic.py`: Heuristic intelligence.
  * `Suggestion` (ABC): Defines a common interface for potential jobs (`OrderSuggestion`, `RelocateSuggestion`).
  * `OrderOptimizer`: Sorts orders by "tightest cluster" efficiency (bounding box of required SKUs).
  * `EdgeAwareOrderScorer`: Estimates the cost to fulfill an order (pick distance + edge travel).

### Validation & Analysis
* `validator.py` (`SubmissionValidator`): Strict, browser-parity rule engine. Simulates the exact state machine step-by-step (`undock -> pick -> dock -> move -> fulfill`). Fails fast on collisions or invalid moves.
* `analysis.py`: Persists run metadata to SQLite (`solution_metadata.db`). Tracks cell usage (heatmaps), bottleneck risks, and SKU flows across runs. Used by the solver to intelligently relocate pallets out of high-traffic lanes.

## Strategic "Pressure Points" & Hacks
* **Docking:** The biggest advantage. Moving pallets closer to hotspots or dragging them along saves massive travel time. Handled via the footprint system in `entities.py`.
* **Space-Time A* Waits:** Robots can explicitly "wait" by staying in place to let another robot pass, natively supported by the 3D numpy reservation tensor.
* **Metadata-Guided Relocation:** The solver uses previous run data (`analysis.py`) to move highly requested SKUs into "low use, high flow" cells, avoiding central travel lanes.
* **Hyperparameters (`config.toml`):** Config controls thresholds like LNS iterations, relocation limits, max plan time, and robot role queues. These are the main "dials" to tweak performance.
* **Hyperparameters (`config.toml`):** Config controls thresholds like LNS iterations, `num_allowed_relocations`, `order_suggestion_gain_constant`, and max plan time. These are the main "dials" to tweak performance.

## Future Change Hints
* **Adding a new heuristic:** Create a new `Suggestion` subclass in `logic.py` and have it generated in `solver.py`'s `_build_suggestion_queue` method.
* **Tweaking pathfinding:** Look at `routing.py`. If robots are getting deadlocked, we may need to adjust the `max_path_steps` or the Space-Time A* cost function.
* **Validating new actions:** Any new mechanics *must* be added to the phased tick pipeline in `validator.py` (`_run_tick`).