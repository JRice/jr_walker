# jr_walker Summary (Compact AI Handoff)

## Scope
- Python warehouse solver for ANE validator format on a 60x40 grid with 5 robots.
- Pipeline: `find_solution` -> optional LNS -> validator replay -> metadata/report write.

## Key Files
- `main.py`: TOML config parse + solver build + run/validate/output.
- `src/jr_walker/solver.py`: core planning logic (legacy suggestion mode + new conveyor mode).
- `src/jr_walker/planner.py` + `src/jr_walker/routing.py`: reservation table + space-time A*.
- `src/jr_walker/validator.py`: action legality and final fulfillment checks.

## Planning Modes
- Legacy mode (default false toggle): suggestion queue (`setup -> relocate -> dock -> order`), retries/backoff, hotspot masks.
- New mode (`solver.single_nest_conveyor_mode=true`):
  - Uses only first configured hotspot, projected to top edge.
  - Builds one 10x3 nest rectangle with 20 pallets:
    - odd SKUs on row 0, even SKUs on row 2.
  - All robots cooperate on setup using nearest source + nearest robot assignment with source fallbacks.
  - Hard integrity check: fail immediately if expected 20 placements are not present in the nest rectangle.
  - Post-build barrier sync + robot exit from nest + dedicated staging move to non-blocking queue cells before conveyor loop.
  - Staging is retry-based (delayed-start retries) and falls back to in-place start if staging move remains invalid.
  - Post-staging barrier sync aligns all robots before order dispatch.

## Conveyor Fulfillment Flow
- Entry: east side of nest row 1.
- Picking: nest-only picks, nearest-needed SKU first, west-only movement during pick/exit stage.
- Wait policy: robot may wait; same-cell wait for `>= conveyor_wait_stall_limit` ticks fails immediately with robot/tick.
- Inventory policy: after leaving nest, order inventory must be complete; missing SKU counts fail immediately.
- Assignment policy: orders are dispatched in strict robot-id round-robin to keep all robots active in the conveyor loop.
- Sticky assignment: each order retains its initially assigned round-robin robot on retries (no cross-robot stealing of failed orders).
- Entry policy: robots now wait/retry for nest entry before requeueing instead of failing fast on first blocked entry path.
- Fulfill policy: forced fulfill cell now uses wait/retry when temporarily blocked before failing.
- Return policy: conveyor return waypoints now use wait/retry and can skip a blocked waypoint after bounded retries (logged), preventing full-run abort on late-loop congestion.
- Delivery/loop path:
  - forced `fulfill` cell is `[hotspot_x-1, hotspot_y]`
  - then loop legs: west 1, south 3, east 12, north 2 (clamped to map bounds)
- Non-fatal order-planning conflicts requeue the order while preserving round-robin progression; solver fails only on hard invariants or prolonged no-progress.

## New Config Fields
- `solver.single_nest_conveyor_mode` (bool)
- `solver.conveyor_wait_stall_limit` (int, default 12)

## Validation/Test Status
- Full unit suite (`unittest discover`) passing after mode addition.
- Smoke run with conveyor mode on a stride-200 subset completed planning/execution flow; expected partial-run validation mismatch remains when input contains more orders than the stride subset.

## Risk Concentration
- `solver.py` complexity is high; fragile areas are now:
  - shared-nest build sequencing under reservation conflicts (now robot-rotated for fairness),
  - conveyor wait/state tracking across retries,
  - strict fail-fast policy interactions with parallel robot timelines.
