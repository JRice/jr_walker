# jr_walker Summary (Compact AI Handoff)

## Scope
- Python warehouse solver for ANE format on a 58x13 map with 5 robots.
- Pipeline: `find_solution` -> optional LNS -> validator replay -> outputs/metadata.
- Legacy suggestion dispatcher path has been removed; solver is conveyor-only now.

## Key Files
- `main.py`: config/load/run/validate/write.
- `src/jr_walker/solver.py`: conveyor planning + setup relocation primitives.
- `src/jr_walker/planner.py`, `src/jr_walker/routing.py`: reservation + space-time A*.
- `src/jr_walker/validator.py`: authoritative action legality check.

## Active Conveyor Design
- Toggle: `solver.single_nest_conveyor_mode=true`.
- `1 hotspot`: single nest conveyor.
- `2+ hotspots`: dual conveyor using first two hotspots.
  - Grouping is fixed `A=[0,1]`, `B=[2,3,4]`.
  - Shared `reserved_source_pallet_ids` prevents both hotspots claiming the same source pallet.
- Nest geometry is `10x3` per hotspot:
  - row `0`: 10 odd-SKU pallets,
  - row `2`: 10 even-SKU pallets.
- Build integrity hard-fails if expected 20 pallets are not in each nest rectangle.

## Config Simplification
- Removed suggestion-related TOML knobs from `docs/config.toml` and `main.py` parser/wiring.
- Kept hotspot config under `[relocation].setup_hotspots` unchanged.

## Conveyor Track + Picking Rules
- Forced fulfill cell: `[hotspot_x - 1, hotspot_y]`.
- Track loop remains fixed: west, south, east, north, re-enter east side.
- Directional picking:
  - moving **west on row 1**: pick from row `0` pallet at same `x`.
  - moving **east on row 3**: pick from row `2` pallet at same `x`.
- After `fulfill()`, robot always advances one track cell (no parking on fulfill cell).
- Orders are planned from the post-fulfill clear cell so each order gets a full loop pass.
- Wait guard: same-cell waits `>= conveyor_wait_stall_limit` fail fast.

## Dispatch Policy
- Orders sorted by `(total picks, sku-count, order_idx)`.
- Single nest: round-robin across all 5 robots.
- Dual nest: weighted group pattern `A,B,B,A,B` with round-robin inside each group.

## Analysis/DB Policy
- Startup no longer reads prior best-solution SQL analysis for planning.
- High-traffic-cell avoidance bootstrap from DB is removed.

## Current Validation Snapshot
- Unit tests: `tests.test_solver_metadata_policy` passing.
- Latest full dual run (current behavior) validated:
  - `output/solution_9023_116.txt`
  - `1000/1000` orders fulfilled
  - makespan `9023`.

## Main Fragile Spots
- `solver.py` remains high-complexity.
- Highest risk is conveyor timing conflicts around shared fulfill/entry chokepoints.
