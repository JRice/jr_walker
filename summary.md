# Codebase Summary (jr_walker)

## 1) What this project is
- A Python solver for the "atoms not electrons" warehouse challenge.
- Objective: minimize makespan (timesteps) for fulfilling all orders with 5 robots on a 60x40 grid.
- Main strategy: greedy suggestion dispatch + space-time A* reservations + strict validator replay + optional LNS post-optimization.

## 2) High-level architecture
- `main.py`
  - Entry point, config loading, run loop, output naming, metadata persistence, and runtime reporting.
- `src/jr_walker/solver.py`
  - Main planning brain (`WarehouseSolver`), suggestion queue, robot assignment, path planning orchestration, retries/backoff, setup/relocation/dock/order execution.
- `src/jr_walker/planner.py` + `src/jr_walker/routing.py` + `src/jr_walker/entities.py`
  - Reservation-based pathfinding stack.
  - `routing.find_path`: Space-Time A* over `(t, x, y)` with WAIT moves.
  - `ReservationPlanner`: wraps pathfinding and reservation table operations.
- `src/jr_walker/scheduler.py`
  - Fast pallet/SKU indexes and greedy pick/fulfill helper methods.
- `src/jr_walker/logic.py`
  - Suggestion classes + heuristic scoring (`OrderSuggestion`, `RelocateSuggestion`, `DockSuggestion`, `SetupSuggestion`, `OrderOptimizer`, `EdgeAwareOrderScorer`).
- `src/jr_walker/validator.py`
  - Browser-parity simulation validator, phased tick execution, and collision legality checks.
- `src/jr_walker/analysis.py`
  - Replay-based metadata extraction and SQLite persistence (`solution_metadata.db`) for cross-run learning.
- `show_metadata.py`
  - Metadata inspection/reporting utilities and tick-0 map rendering.

## 3) End-to-end runtime flow
- `main.py`:
  - Loads `docs/config.toml` into `RunConfig`.
  - Loads prior best-run analysis from SQLite (`load_best_past_analysis`).
  - Builds solver via `_build_solver(...)`.
  - Calls:
    - `solver.find_solution()` (dispatch planner)
    - `solver.optimize_actions()` (LNS pass)
    - `SubmissionValidator` full replay check
  - Writes output actions and stores metadata (`build_and_store_solution_metadata`).
  - Prints total runtime and rolling progress messages.

## 4) "Meat" of the logic (most important files/methods)
- Primary hotspot: `src/jr_walker/solver.py` (3272 lines).
- Core control loop:
  - `_find_solution_actions_core` (line ~586)
  - Greedy dispatch loop over a mixed suggestion queue.
- Setup system:
  - `_build_setup_jobs` (line ~1032)
  - `_setup_slot_candidates` (line ~925)
  - `_plan_setup_pallet_for_robot` (line ~2085)
  - `_attempt_relocation_via_stand` (line ~2217)
- Suggestion generation and reprioritization:
  - `_build_suggestion_queue` (line ~814)
  - `_refresh_pending_order_suggestions` (line ~1135)
- Robot selection/policy:
  - `_candidate_robots_for_suggestion` (line ~1750)
  - `_setup_probe_robot_for_job` (line ~1785)
- Path/commit safety layer:
  - `_safe_plan_path`, `_safe_plan_path_with_step_cap` (line ~2732, ~2773)
  - `_can_commit_pending_actions`, `_commit_plan` (line ~2822, ~2837)
- Post-processing/repair:
  - `_repair_idle_wait_conflicts` (line ~3120)
  - `_lns_improve_actions`, `_lns_shift_candidate` (line ~2895, ~2939)

## 5) Algorithms and strategies currently in use

### 5.1 Dispatch strategy
- Greedy suggestion dispatcher:
  - Builds setup + optional relocation + dock + order suggestions.
  - Sort key is mostly `gain - cost` with hard priority for setup suggestions.
  - Retries failed non-setup suggestions with exponential backoff and drop limit.
  - Retries failed setup suggestions by waiting `setup_retry_wait_ticks` (makespan-time gate), then drops at retry limit.
  - Per-robot fail streak can trigger parking moves to reduce blocking.
  - If a DockSuggestion throws `ValueError("Cannot reserve footprint for robot=...")`, it is skipped (fail-soft) instead of failing the run.

### 5.2 Setup strategy (recently evolved)
- Preplanned hotspot-oriented setup jobs:
  - Edge-anchored two-line template:
    - odd SKUs on the edge line from hotspot in ascending order,
    - even SKUs on the line two cells inward, also ascending.
- Same-column frontier gating:
  - A setup job only runs when all earlier slot jobs for that hotspot are completed/dropped.
- Dedicated setup robot by hotspot:
  - `_assign_setup_robots_by_hotspot` maps one robot per hotspot when possible.
- Locality-biased source reservation:
  - nearest source pallet per hotspot/SKU is biased to pallets "owned" by that hotspot region, reducing cross-hotspot source stealing.
- Reachability-first fallback ranking:
  - `_setup_probe_robot_for_job` probes path feasibility to stand cells and prioritizes reachable robots.
- Foreign-corridor stand avoidance:
  - setup stand cells are prioritized away from other hotspots' active setup corridors while they still have pending setup jobs.
- Reorientation for tight setup moves:
  - In `_attempt_relocation_via_stand`, setup jobs can do pull -> undock -> edge-side reposition -> redock before final carry.
  - Includes limited pull-round retries for blocked micro-moves.

### 5.3 Order strategy
- `OrderOptimizer.find_tightest_cluster`:
  - Picks an anchor SKU (rarest first), greedily picks nearest pallets for other SKUs, scores by bounding-box perimeter.
- `OrderSuggestion`:
  - Uses fixed gain constant and a cheap cost estimate from cluster span + edge distance.
  - Carries preferred per-SKU pallet cells from the computed cluster.
- Pick execution policy:
  - try suggestion-provided preferred pallet cells first,
  - if not feasible, fall back to nearest-available pick options (previous behavior).
- `EdgeAwareOrderScorer` exists for richer detour-aware estimation but current dispatch queue mainly uses `OrderSuggestion` scoring.

### 5.4 Dock strategy
- `DockSuggestion` generation from historical fulfill streaks (`_build_dock_suggestions`).
- Looks for repeated SKU streaks by robot in prior run metadata.
- Gain is scaled by `dock_gain_scale`, with `min_jobs_for_dock` gating short streaks.

### 5.5 Relocation strategy
- Optional (`enable_relocation_suggestions` default false).
- Metadata-guided placement:
  - Bucket-lift heuristic + SKU anchor rows from `cell_sku_flow`.
  - Avoids travel lanes and high-use cells where possible.
  - Candidate scoring penalizes choke points and density.

### 5.6 Pathfinding and collision model
- Space-Time A* over reservation tensor (3D `[time, y, x]`):
  - Supports WAIT moves.
  - Footprint includes robot + all docked pallets.
- Planner reserves both paths and single-timestep footprints for non-move actions.
- Candidate plans are validator-checked before commit.

### 5.7 Validation model
- Validator executes in strict phased order each tick:
  - `undock -> pick -> dock -> move -> fulfill`
- Includes checks for:
  - move-to-same-cell collisions,
  - moving into stationary robots,
  - docked pallet projected collisions (`G2`, `H2` cases),
  - legality constraints per action.

### 5.8 Metadata/learning loop
- `analysis.py` replays each solution to compute:
  - cell use/collision risk/picks/travel time,
  - edge fulfills + SKU flow,
  - robot idle/empty-move/order-time stats.
- Stored in SQLite and consumed next run by solver for lane/high-traffic avoidance and relocation targeting.

## 6) Most complex methods (risk + maintenance cost)
- `solver._attempt_relocation_via_stand` (~246 lines)
  - Highest branching depth.
  - Handles dock/carry/undock + setup-specific reorientation logic + retry loops.
- `solver._find_solution_actions_core` (~228 lines)
  - Global orchestrator with queue mutation, retries/backoff, robot ranking, and completion bookkeeping.
- `solver._repair_idle_wait_conflicts` (~153 lines)
  - Post-hoc repair pass with custom simulation-lite logic.
- `solver._plan_setup_pallet_for_robot` (~92 lines)
  - Multi-source fallback, stand search, reason accounting, source-swapping.
- `validator._run_tick` + related validators
  - Critical correctness surface; many invariants coupled across helper methods.

## 7) Opaque or fragile areas
- Setup relocation internals are hard to reason about:
  - `_plan_setup_pallet_for_robot` + `_attempt_relocation_via_stand` combine pathfinding, collision safety, and dynamic source/target rebinding.
- Dual-state coupling:
  - Pallet truth exists across `scheduler.pallets`, `pallet_by_id`, `pallet_id_by_coord`, and reservation/static obstacle layers.
  - Desync risk is real if one update path is missed.
- Repair step complexity:
  - `_repair_idle_wait_conflicts` can mask upstream planning issues; good safety net, but hard to validate mentally.
- Heuristic overlap:
  - `OrderSuggestion` and `EdgeAwareOrderScorer` coexist; queue currently favors the simpler cost model, so intent can be unclear to new contributors.
- Static obstacle timing:
  - `can_add_static_obstacle_from(...)` and delayed static insertion after undock are subtle but critical for avoiding false conflicts.

## 8) Best tuning levers

### 8.1 Speed/runtime tuning
- A* pressure:
  - `solver.path_step_limit`
  - `solver.relocate_stand_candidate_limit`
  - `solver.relocate_target_candidate_limit`
  - `solver.max_robots_per_suggestion`
- Retry churn:
  - `solver.suggestion_retry_limit`
  - `solver.suggestion_backoff_base_cycles`
  - `solver.suggestion_backoff_max_cycles`
  - `solver.setup_retry_wait_ticks` (setup-only retry wait gate)
- Validation overhead:
  - `solver.ticks_to_full_validation` (periodic full replay interval)
- LNS budget:
  - `[lns].iterations`, `window_actions`, `tail_fraction`, `max_shift`
- Order refresh frequency:
  - now refreshed after every completed setup/relocation; improves responsiveness but adds recompute overhead.

### 8.2 Makespan-quality tuning
- Setup packing quality:
  - `relocation.setup_hotspots`
  - `relocation.setup_hotspot_radius`
  - `_setup_slot_candidates` pattern and hotspot assignments.
- Dispatch economics:
  - `solver.order_suggestion_gain_constant`
  - `solver.dock_gain_scale`
  - `solver.relocation_gain_scale`
  - `solver.min_jobs_for_dock`
- Congestion management:
  - `relocation.lane_width`
  - `relocation.edge_band_for_heatmap`
  - `relocation.relocate_chunk_size`
- Failure handling behavior:
  - parking thresholds (`robot_fail_streak_for_parking`, `parking_candidate_limit`)
  - setup retry wait (`setup_retry_wait_ticks`) + retry limit.

## 9) Current observability quality
- Strong logging exists in solver:
  - setup progress by hotspot,
  - retry/drop reasons,
  - A* slow/blocked summaries,
  - periodic validation checkpoints,
  - dispatch progress with runtime.
- New useful setup/debug logs:
  - `setup_suggestion_rejected` with robot/hotspot/source/target and detailed rejection reason.
  - `setup_stand_corridor_penalty` showing when foreign-corridor stand penalties applied.
  - `setup_source_reservations` and per-hotspot `cross_hotspot_sources`.
  - crash context now includes `Active suggestion at failure: ...` in `main.py`.
- Good foundation for debugging setup gaps:
  - failure reason counters are already emitted (`setup_attempt_fail ... failure_reasons=...`).
- Remaining gap:
  - still no concise end-of-run histogram that aggregates setup failure reasons by hotspot/robot/source.

## 10) Tests and what they protect
- `tests/test_solver_metadata_policy.py`
  - setup candidate/robot policy behavior,
  - strict-no-swap check,
  - metadata targeting behavior.
- `tests/test_analysis_metadata.py`
  - metadata schema/path/run-id selection and core metric correctness.
- `tests/test_planner_reservations.py`
  - static obstacle insertion conflict logic.
- `tests/test_validator_move_shape_conflicts.py`
  - projected move-shape collision cases (`G2`, `H2`).

Coverage is targeted around high-risk policy logic and validation invariants, not broad.

## 11) Practical map for future AI tasks
- If task is "why robot chose X suggestion": start in `solver._find_solution_actions_core`.
- If task is "setup gaps/jagged columns": inspect
  - `_build_setup_jobs`,
  - `_setup_slot_candidates`,
  - `_setup_frontier_blocking_pallet_id`,
  - `_plan_setup_pallet_for_robot`,
  - `_attempt_relocation_via_stand`,
  - logs in `output/run.log`.
- If task is "dock never selected":
  - `_build_dock_suggestions`,
  - suggestion sort/scoring,
  - candidate robot filtering in `_candidate_robots_for_suggestion`.
- If task is "collisions/invalid output":
  - `validator.py` first, then `_can_commit_pending_actions` and repair pass.
- If task is "slow run":
  - inspect `astar_summary`, periodic validation interval, and suggestion retry churn.

## 12) Suggested medium-term refactors
- Split `_attempt_relocation_via_stand` into smaller state-machine steps.
- Extract a shared "plan candidate + validate + commit" helper to reduce duplication across order/setup/dock/relocate.
- Consolidate pallet-state update paths into one authoritative helper to reduce desync risk.
- Decide on one primary order-cost model (simple cluster vs edge-aware scorer) and wire it consistently.
- Add a run-end setup diagnostics block (per-hotspot completions, drops, fail reasons, average A* cost per setup move).
