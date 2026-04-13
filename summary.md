# jr_walker Summary (Condensed, AI-Oriented)

## 1) Goal + Shape
- Python solver for the Atoms Not Electrons warehouse task.
- Optimize makespan for 5 robots on a 60x40 grid while maintaining validator-legal actions.
- Core approach:
  - greedy suggestion dispatch,
  - space-time reservation path planning,
  - validator-compatible commit checks,
  - optional LNS post-optimization.

## 2) Primary Files
- `main.py`
  - Loads `docs/config.toml`, builds `SolverConfig`, runs solve + optimize + validation + metadata write.
- `src/jr_walker/solver.py`
  - Main orchestrator (`WarehouseSolver`): setup jobs, suggestion queue, robot assignment/ranking, path/commit logic.
- `src/jr_walker/logic.py`
  - Suggestion types and order heuristics (`OrderSuggestion`, `RelocateSuggestion`, `DockSuggestion`, `SetupSuggestion`, `OrderOptimizer`).
- `src/jr_walker/planner.py` + `src/jr_walker/routing.py`
  - Reservation-based pathing (space-time A* with WAIT).
- `src/jr_walker/scheduler.py`
  - SKU/pallet indexes and pick-cell candidate utilities.
- `src/jr_walker/validator.py`
  - Browser-parity legality model and phased tick execution.
- `src/jr_walker/analysis.py`
  - Replay metadata into SQLite for future-run guidance.

## 3) End-to-End Runtime
1. Parse TOML into `RunConfig` (`main.py`).
2. Load prior metadata (`load_best_past_analysis`).
3. Build solver and run `find_solution()`.
4. Optional LNS `optimize_actions()`.
5. Full validator replay.
6. Write output + metadata.

## 4) Current Dispatch Policy (Important)
- Suggestion queue composition:
  - setup suggestions first,
  - optional relocation suggestions,
  - optional dock suggestions,
  - order suggestions.
- Retry model:
  - non-setup suggestions: exponential backoff + drop at retry limit.
  - setup suggestions: retry wait gate (`setup_retry_wait_ticks`) + drop at retry limit.
- Parking:
  - repeated failures can trigger idle parking moves.

## 5) Setup Phase Behavior
- Hotspot-driven setup jobs build tight two-line SKU templates near each setup hotspot.
- One setup robot can be assigned per hotspot (`_assign_setup_robots_by_hotspot`).
- Setup jobs use frontier gating to preserve intended fill order.

## 6) Order Strategy (Latest)
- Order suggestion generation now supports two modes:
  - Legacy cluster mode (`enable_order_cluster_discovery=true`): uses `OrderOptimizer.find_tightest_cluster`.
  - Default mode (`false`): no cluster discovery; order suggestions are generated once and kept stable.
- Order suggestion refresh during dispatch is configurable:
  - `recalculate_order_suggestions` (default `false`).

### Pick candidate policy (latest)
For each needed SKU in an order:
1. Try suggestion-provided preferred pallet cells (if any).
2. Otherwise run BFS-based candidate discovery from an origin:
   - local search up to `order_pick_local_manhattan_radius` (default `10`),
   - if no viable candidate, fallback to global search.
3. Candidate scoring can penalize picks too close to non-assigned hotspots:
   - `order_other_hotspot_penalty`,
   - `order_other_hotspot_penalty_radius`.

## 7) Hybrid Hotspot Assignment (Latest)
- Robots explicitly assigned during setup keep a persistent `robot.assigned_hotspot` for the full run.
- Robots not assigned during setup remain unassigned:
  - their order-pick BFS origin is their current live `(x, y)` (dynamic).
- This enables mixed behavior: sticky hotspot robots + free-roaming robots.

## 8) Safety/Correctness Model
- Plans are only committed after reservation legality checks.
- Validator model phases each tick:
  - `undock -> pick -> dock -> move -> fulfill`.
- Collision/legality protections include move conflicts, footprint conflicts (robot + docked pallets), and action invariants.

## 9) Most Useful Config Knobs (Current)
- Order/hotspot knobs:
  - `solver.enable_order_cluster_discovery`
  - `solver.recalculate_order_suggestions`
  - `solver.order_pick_local_manhattan_radius`
  - `solver.order_other_hotspot_penalty`
  - `solver.order_other_hotspot_penalty_radius`
- Dispatch churn knobs:
  - `solver.suggestion_retry_limit`
  - `solver.suggestion_backoff_base_cycles`
  - `solver.suggestion_backoff_max_cycles`
  - `solver.max_robots_per_suggestion`
- Setup/traffic knobs:
  - `relocation.setup_hotspots`
  - `solver.setup_retry_wait_ticks`
  - `solver.robot_fail_streak_for_parking`
  - `solver.parking_candidate_limit`
- Perf knobs:
  - `solver.path_step_limit`
  - `solver.ticks_to_full_validation`
  - `[lns]` section (`iterations`, `window_actions`, `tail_fraction`, `max_shift`)

## 10) Fast Debug Entry Points
- “Why this suggestion/robot?”
  - `_find_solution_actions_core`
  - `_candidate_robots_for_suggestion`
- “Order picking weird / hotspot drift?”
  - `_plan_order_for_robot`
  - `_bfs_pick_candidates_for_remaining`
  - `_robot_assigned_hotspot`
- “Setup misses / setup stalls?”
  - `_build_setup_jobs`
  - `_plan_setup_pallet_for_robot`
  - setup frontier helpers
- “Validation/collision failure?”
  - `validator.py`
  - `_can_commit_pending_actions`
  - reservation planner methods

## 11) Tests Covering Recent Risk
- `tests/test_solver_metadata_policy.py` now includes:
  - local-radius BFS -> global fallback behavior,
  - hotspot proximity penalty behavior,
  - hybrid hotspot assignment behavior (persistent only for initially assigned robots),
  - unassigned robot origin uses live position.
- Other suites continue to guard validator/planner/metadata invariants.

## 12) Known Complexity Hotspots
- `solver.py` remains the highest cognitive-load file (dispatch + setup + commit orchestration).
- Setup relocation internals are still the most branch-heavy area.
- Multiple pallet-state representations require careful synchronization.
