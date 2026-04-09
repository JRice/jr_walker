# atoms not electrons

Congratulations, you have inherited your great uncle's warehouse. The Big Order is due and the Client is breathing down your neck. Fulfill it as fast as possible by telling your robot fleet what to do. Good luck!

## The Challenge

You command **5 robots** in a **60x40 grid** warehouse. Your mission: fulfill **1,000 orders** by collecting items from pallets and delivering them to the fulfillment zone.

**Your score** = total timesteps to complete ALL orders. Lower is better.

## Key Concepts

| Term | Meaning |
|------|---------|
| **Pallet** | A container holding infinite items of a single SKU type (e.g., "Pallet at (3,5) has SKU 7") |
| **Order** | A list of SKUs to collect and deliver together (e.g., "2x SKU 1, 1x SKU 3, 1x SKU 7") |
| **Storage** | Robot's internal inventory where picked items are held |
| **Fulfill** | Deliver items at the perimeter (x=0, x=59, y=0, or y=39) when your storage EXACTLY matches an unfulfilled order |

## The Grid

- **Dimensions**: 60 cells wide x 40 cells tall, indexed (x,y) from (0,0) at top-left
- **Fulfillment zone**: Entire perimeter (x=0, x=59, y=0, or y=39) — robots deliver orders here

## Robot Actions

Each robot executes AT MOST ONE action per timestep:

| Action | Usage | Effect |
|--------|-------|--------|
| `move` | `move <x> <y>` | Move to an adjacent empty cell |
| `pick` | `pick <x> <y>` | Pick 1 item from an adjacent pallet into storage |
| `dock` | `dock <x> <y>` | Attach to an adjacent pallet (see Docking below) |
| `undock` | `undock <x> <y>` | Detach from a docked pallet |
| `fulfill` | `fulfill <x> <y>` | Deliver order at perimeter (coords ignored) |

## Docking Explained

**Why dock?** Normally, robots must be adjacent to a pallet to pick from it. By docking, you attach a pallet to your robot — they move together as a unit.

**Use cases:**
- Move a pallet closer to the fulfillment zone
- Carry a frequently-needed pallet with you instead of returning to it
- Reposition pallets to reduce travel time

**How it works:**
- A docked pallet moves WITH the robot (same direction, same timestep)
- Docked pallets still occupy grid cells and can collide with other entities
- Robots can dock up to 4 pallets (one on each side)
- You can still pick from docked pallets (yours or others')

## Rules

1. **Movement**: Adjacent cells only (not diagonal). Target must be empty.
2. **Picking**: Pallets never run out. Multiple robots can pick from the same pallet.
3. **Fulfillment**: Storage must EXACTLY match an unfulfilled order — no more, no less.
4. **Collision**: No two entities can occupy the same cell.

## Submission Format

Text file with one action per line:
```
<timestep> <robot_id> <action> <x> <y>
```

Example:
```
0 0 move 1 0
0 1 pick 10 5
1 0 move 2 0
1 0 pick 3 0
2 0 move 59 0
3 0 fulfill 0 0
```

Requirements:
- Lines must be in increasing timestep order
- No duplicate (timestep, robot_id) pairs
- Robots with no action at a timestep simply wait

## Worklist Format (BIG_ORDER.txt)

```
<num_robots>
<x> <y>              # Robot starting positions
...
<num_pallets>
<x> <y> <sku>        # Pallet positions and SKU types
...
<num_orders>
<sku1> <sku2> ...    # Space-separated SKUs per order
...
```

## The Big Order

- **5** robots
- **20** unique SKU types
- **240** pallets (multiple pallets may have the same SKU)
- **1,000** orders (~10 items each on average)
- **SKU distribution**: Power law (Zipf-like) — some SKUs are "high runners" appearing much more frequently in orders

## How to Participate

1. **Download** `BIG_ORDER.txt` — the warehouse state and orders
2. **Write a solver** that outputs robot commands. AI agent use strongly recommended.
3. **Use the Testbench** — upload your solution to visualize and debug before submitting
4. **Submit to leaderboard** — requires GitHub login

## Testbench

The testbench lets you validate and debug your solution before submitting.

1. Click **TESTBENCH** on the homepage
2. Drop your `.txt` solution file
3. The simulation runs in-browser — scrub the timeline to step through each timestep
4. Inspect robot positions, pallet contents, and order fulfillment
5. When ready, click **Submit to Leaderboard** (requires GitHub login)

All validation happens locally. Nothing touches the server until you explicitly submit.

## Tips

- Plan efficient paths — minimize robot travel time
- Consider which robots are closest to which pallets
- Docking can reduce repeated trips to distant pallets
- Multiple robots can work in parallel
- Batch orders by shared SKUs
- Start simple with a single-robot no-docking naive solution, then optimize

---

*AI use encouraged. Don't spam. Play fair.*

*Created with love at [Tutor Intelligence](https://tutorintelligence.com), building generally capable robot workers for American industry. If you enjoy this puzzle, please [consider joining our team](https://jobs.lever.co/tutorintelligence)!*