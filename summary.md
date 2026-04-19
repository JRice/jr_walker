# jr_walker Summary (Compact AI Handoff)

## Scope
- Python warehouse solver for Atoms Not Electrons.
- Target: low makespan with validator-legal actions on a 60x40 grid, 5 robots.
- Pipeline: solve (`WarehouseSolver.find_solution`) -> optional LNS optimize -> full validator replay -> metadata write.

## Key Files
- `main.py`: config load, run orchestration, validation, output/metadata.
- `src/jr_walker/solver.py`: main dispatch/setup/order/path orchestration.
- `src/jr_walker/logic.py`: suggestion types and ranking heuristics.
- `src/jr_walker/planner.py`, `src/jr_walker/routing.py`: reservation planner + A* pathing.
- `src/jr_walker/validator.py`: legality model (`undock -> pick -> dock -> move -> fulfill`).
- `check_churn.py`: run-log tail churn analyzer (added this chat).

## Dispatch/Planning Model
- Suggestion queue priority: setup -> relocation (optional) -> dock (optional) -> order.
- Retries: exponential backoff for non-setup; setup has wait-gated retry.
- Commit guard: reserve/check first, then append actions.
- Persistent hotspot assignment is hybrid:
  - setup-assigned robots keep `assigned_hotspot`,
  - non-assigned robots remain free-roaming with dynamic pick origin.

## Setup Behavior (Current)
- Setup jobs form edge-anchored two-line hotspot templates.
- Optional dual setup relocation (`solver.enable_setup_dual_relocation`):
  - one robot can build two-block base pairs in one plan (`_plan_setup_pair_for_robot`),
  - partner pairing via `_pending_setup_pair_job`,
  - target mapping via `_setup_pair_target_for_job`,
  - geometric validation via `_setup_pair_geometry`.
- Pair-shaping now supports all edges, with N/S-specific clearance:
  - before pivoting, N/S hotspots can pull inward up to 2 cells while docked (`_setup_pair_pull_clearance`),
  - then undock/re-dock pivot (`_execute_local_pivot_maneuver`) to reach the required I-shape.

## Changes Added In This Chat
- Stuck docked robot recovery:
  - added `_plan_idle_recovery_undock` and dispatch fallback hook so a robot that delivered once and stalled can undock and rejoin work.
- Setup pair relocation improvements:
  - enabled two-block base construction in one combined plan,
  - generalized pair logic beyond west edge,
  - added N/S pull-clearance step for tightly packed north/south rows.
- Non-hotspot A* masking refinement:
  - separate A*-specific forbidden mask for non-hotspot robots,
  - for top/bottom hotspots, mask outside two rows (not full 3-row protected band) so pass-through routing is less overblocked.
- Churn diagnostics:
  - new root script `check_churn.py` parses tail of run log (`--lines`, default 120),
  - reports churn score and components including `astar_blocked` ratio and repeated target/signature attempts.

## Practical Debug Entry Points
- Suggestion/robot choice: `_find_solution_actions_core`, `_candidate_robots_for_suggestion`.
- Order pick drift/churn: `_plan_order_for_robot`, `_bfs_pick_candidates_for_remaining`, `_other_hotspot_proximity_penalty`.
- Setup stalls/pair shaping: `_plan_setup_pallet_for_robot`, `_plan_setup_pair_for_robot`, `_setup_pair_pull_clearance`.
- Path blocking: `_forbidden_cells_for_robot`, hotspot mask builders, `_safe_plan_path*`.
- Recovery behavior: `_plan_idle_recovery_undock`.

## Tests
- `tests/test_solver_metadata_policy.py` includes coverage for:
  - hotspot assignment hybrid policy,
  - local BFS -> global fallback,
  - hotspot proximity penalty,
  - non-hotspot A* mask behavior,
  - setup pair geometry/targeting and N/S clearance direction,
  - idle undock recovery behavior.

## Current Risk Concentration
- `solver.py` remains the highest-complexity file.
- Most fragile areas: setup pair maneuvers, forbidden-cell masking, and pallet/robot state synchronization across staged plans.
