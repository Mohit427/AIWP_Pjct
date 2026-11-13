# Memory-Bounded Search, Visualization Interface, and Model Limitations

This document provides the report sections covering:
1. **Algorithms Used**: Iterative Deepening A* (`IDA*`), Depth-First Branch and Bound (`DFBB`), and `Random Walk` (local search baseline).
2. **Implementation and Visualization Interface**: How `ev/viz.py` and `main.py` render route plans, step-by-step search playback, memory comparisons, and charger placement optimizations from the standard `Result` and `LocalSearchResult` records.
3. **Limitations and Algorithms Left Out on Purpose**: Honest engineering limitations of the discrete EV state-space model and theoretical justification for excluding adversarial search and constraint satisfaction algorithms.

---

## 1. Algorithms Used: `IDA*`, `DFBB`, and `Random Walk`

### 1.1 Iterative Deepening A* (`IDA*` — `ev/memory_bounded.py`)
- **Design & Formulation**: `IDA*` conducts a sequence of depth-first searches bounded by an increasing $f$-cost threshold $T_k$, expanding only states satisfying $f(s) = g(s) + \lceil h(s) \rceil \le T_k$. Because every road travel time (`Road.minutes`) and unit charging time (`Charger.minutes_per_unit`) is a positive integer ($\ge 1$ minute), the true optimal remaining time $h^*(s)$ is always an integer. Taking the ceiling $\lceil h(s) \rceil$ of any admissible heuristic $h(s) \le h^*(s)$ preserves both admissibility ($\lceil h(s) \rceil \le h^*(s)$) and edge consistency while keeping $f$-thresholds on integer minute boundaries.
- **Handling Graph Cycles & Charging Loops**: On cyclic road networks with multi-unit charging stations, naive tree-search `IDA*` suffers from severe threshold thrashing when dozens of intermediate paths revisit the same `(node, battery)` states across slightly higher integer thresholds. Our implementation keeps an explicit $O(b \cdot d)$ active path stack (`on_path` cycle avoidance) combined with admissible Bellman subtree lower-bound backups (`learned_h[s] = min_sub_f - g(s)` whenever a subtree under $s$ finishes without reaching the goal). By induction on the Bellman inequality, `learned_h[s]` never overestimates $h^*(s)$, allowing `IDA*` to solve the 150-node `large` map in 11,217 expansions (88 ms) with a peak frontier of only **10 states** (compared to 96 states for `A*`).
- **Completeness, Optimality & Complexity**:
  - **Complete**: Yes on finite state spaces with step costs $\ge 1$.
  - **Optimal**: Yes; returns the exact same minimum trip minutes as `UCS` and `A*` across all 5 standard maps (`toy`: 70 min, `corridor`: 437 min, `small`: 48 min, `medium`: 329 min, `large`: 577 min) and all 90 seeded start–goal experiment pairs.
  - **Memory**: $O(b \cdot d)$ active frontier stack (e.g., peak frontier of 6 states and 32.1 KB peak memory on `corridor`, vs. 71 states and 53.2 KB for `A*`).

### 1.2 Depth-First Branch and Bound (`DFBB` — `ev/memory_bounded.py`)
- **Design & Formulation**: `DFBB` performs a single depth-first traversal while maintaining an incumbent upper bound $U$ (`best_cost`, initialized either to $\infty$ or to a caller-supplied `initial_bound`). Any branch with $f(s) = g(s) + \lceil h(s) \rceil \ge U$ is pruned immediately, and whenever a goal state is reached with $g(\text{goal}) < U$, the incumbent bound is tightened to $U = g(\text{goal})$.
- **Effect of Starting Bound (`DFBB` vs. `DFBB(unbounded)`)**: As predicted by theory, `DFBB` is only as fast as its starting bound:
  - When `use_greedy_bound=True` (or when seeded with an initial greedy bound), `DFBB` orders children by smallest $h(s)$ until the first complete path reaches the destination, rapidly establishing a tight upper bound $U$ before switching to $f(s)$ ordering to prove optimality.
  - On the `medium` map (optimal cost 329 min), seeding `DFBB` with the greedy bound (346 min) cuts node expansions from **3,381 down to 761** and reduces peak memory from **65.7 KB to 42.5 KB**. On the `large` map (optimal cost 577 min), a good starting bound (681 min) cuts expansions from **30,597 down to 7,713** and reduces peak memory from **170.1 KB to 90.3 KB**.

### 1.3 Random Walk Baseline (`random_walk` — `ev/memory_bounded.py`)
- **Design & Role**: Implements the unguided stochastic baseline for both sub-problems:
  1. **Charger Placement (`ChargerPlacementProblem`)**: Starts from a random set of $N$ charger sites out of $M$ candidates and, at each iteration, swaps one chosen site with one uniformly chosen unchosen candidate unconditionally (ignoring whether average trip time improves or worsens), while recording the best configuration encountered in `LocalSearchResult`.
  2. **Route Search (`Problem`)**: Uses an internal seeded generator (`random.Random(seed)`) per `CONTRACT.md` Rule 5 so timed and memory passes inside `measure()` execute identical steps, returning `CUTOFF` whenever the walk hits its step limit or a non-start dead end without proving unsolvability.

### 1.4 Empirical Comparison Summary (`IDA*` and `DFBB` vs. `UCS` and `A*`)

| Map (`cap`) | Optimal Cost (min) | `UCS` Expanded / Frontier / Mem | `A*(H2)` Expanded / Frontier / Mem | `DFBB` Expanded / Frontier / Mem | `IDA*(H2)` Expanded / Frontier / Mem |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`toy` (cap 3)** | **70** | 6 / 5 / 5.0 KB | **4** / 5 / 6.3 KB | 4 / 5 / 5.9 KB | 13 / **1** / 5.7 KB |
| **`corridor` (cap 10)** | **437** | 176 / 39 / 59.2 KB | **118** / 71 / 53.2 KB | 618 / 44 / 35.4 KB | 1,173 / **6** / **32.1 KB** |
| **`small` (cap 10, 10 trips)** | **106.1** (mean) | 75.3 / 39.0 / 32.7 KB | **30.1** / 33.3 / 21.6 KB | 110.9 / 23.6 / 14.0 KB | 139.5 / **3.0** / **9.5 KB** |
| **`medium` (cap 10, 10 trips)** | **360.5** (mean) | 351.2 / 53.5 / 121.7 KB | **228.3** / 70.9 / 95.6 KB | 4,593.8 / 78.5 / 69.1 KB | 3,633.6 / **6.2** / **42.8 KB** |
| **`large` (cap 10, 10 trips)** | **640.5** (mean) | 816.7 / 77.0 / 256.8 KB | **597.9** / 101.7 / 204.6 KB | 41,478.2 / 215.5 / 157.1 KB | 18,755.0 / **11.5** / **102.0 KB** |

---

## 2. Implementation and Visualization Interface (`ev/viz.py` & `main.py`)

To make every search algorithm's behavior transparent without coupling core solvers to external graphics packages, all plotting code is isolated inside `ev/viz.py` (`matplotlib`), while both `main.py` and `ev/viz.py` consume the shared `Result` and `LocalSearchResult` data structures:

1. **Screen 1 — Route Map & Battery State-of-Charge Profile (`render_route_map` / `results/visuals/route_map_corridor.png`)**:
   - Replays `Result.path` through `Problem.successors()` via `reconstruct_route()` to verify legality and extract the exact battery level and cumulative elapsed minutes at every city.
   - Left panel plots the spatial road network, road durations/battery costs, charging stations, start/destination markers, directional arrows along the chosen highway corridor, and callout badges (`+8u (80m)` at Vellore, `+1u (10m)` at Hosur) at active charging stops.
   - Right panel plots the step-by-step battery state-of-charge trajectory against elapsed trip time, illustrating how `A*`/`UCS`/`IDA*`/`DFBB` arrive at intermediate chargers and the final destination with zero wasted charge (`0/10` units at Vellore and Bengaluru).
2. **Screen 2 — Step-by-Step Search Playback (`render_search_playback` / `results/visuals/search_playback_corridor.png`)**:
   - Reads `Result.expanded_order` (the ordered sequence of `(node, battery)` states popped from the frontier) and renders four progressive snapshots at **25%, 50%, 75%, and 100%** of total expansions.
   - Node color encodes the first expansion step index (`plasma` colormap) and marker size scales with the number of distinct battery states expanded at that city, making the directional focus of `A*(H2)` visually distinct from the wide radial sweep of `BFS`/`UCS`.
3. **Screen 3 — Charger Placement Before-and-After View (`render_placement_comparison` / `results/visuals/charger_placement_corridor.png`)**:
   - Compares the initial random placement of $N$ chargers (`Before`) against the local-search optimized placement (`After`) alongside sample trip desire lines and the iteration-by-iteration objective convergence curve from `LocalSearchResult.trace_history`.
4. **Screen 4 — Memory & Frontier Comparison (`render_memory_comparison` / `results/visuals/ida_dfbb_memory_comparison.png`)**:
   - Side-by-side bar charts comparing `UCS`, `A*`, `DFBB`, and `IDA*` across `corridor`, `small`, `medium`, and `large` maps on Peak Frontier Size and Peak Memory (KB).

---

## 3. Limitations and Algorithms Left Out on Purpose

### 3.1 Honest Engineering Limitations of the Model
While our discrete state-space formulation `State = (node, battery)` makes completeness and optimality provable, it introduces five simplifying assumptions relative to real-world EV navigation:

1. **Discretized Battery Units and Conservative Range Rounding**:
   - Battery charge is modeled as an integer from `0` to `MAX_BATTERY_CAP = 10` (where 1 unit $\approx 30\text{ km}$), and `battery_cost_for_km(km)` rounds road energy consumption **up** via `max(1, ceil(km / 30))`.
   - *Impact*: Rounding up guarantees safety (a vehicle never runs out of charge mid-edge), but a 32 km road and a 59 km road both consume 2 battery units. Over a multi-hop route, cumulative ceiling rounding can overestimate energy consumption and force an extra charging stop that a continuous kWh model would avoid.
2. **Linear Charging Rate Assumption (Ignoring CC–CV Tapering)**:
   - Each charging action adds $+1$ battery unit in a constant `minutes_per_unit` regardless of whether the battery is charging from $0 \to 1$ or from $9 \to 10$.
   - *Impact*: Real lithium-ion EV batteries follow a Constant-Current / Constant-Voltage (CC–CV) curve where charging slows down substantially above 80% state-of-charge. Our optimal plans already naturally avoid charging to 100% unless necessary (because any excess battery at the destination costs charging time without reducing driving time), but modeling higher `minutes_per_unit` for units $9$ and $10$ would further penalize top-of-pack charging.
3. **Zero Charger Queuing and Unlimited Stall Availability**:
   - The model assumes a vehicle can plug in immediately upon arriving at any charger node.
   - *Impact*: On busy highway corridors (such as Chennai–Bengaluru on weekends), popular fast chargers (e.g., Vellore or Krishnagiri at 10 min/unit) may have waiting queues, making a slightly slower 15 min/unit station with open stalls faster in practice.
4. **Static Road Travel Times and Flat Topography**:
   - Edge travel times (`Road.minutes`) and energy costs (`Road.battery_cost`) are static constants and symmetric (`roads[a][b] == roads[b][a]`).
   - *Impact*: Real EV energy consumption is asymmetric on elevation gradients (climbing the Eastern Ghats toward Palamaner/Kolar/Hosur consumes more energy than descending toward Chennai, where regenerative braking recovers charge) and varies with live traffic congestion, ambient temperature, and highway speed choice.
5. **Single-Objective Time Optimization vs. Monetary Cost**:
   - Route search minimizes total trip duration (`minutes`), computing monetary charging cost (`price_per_unit` in rupees) post-hoc as a secondary metric. A driver willing to spend 15 extra minutes to save Rs 50 at cheaper chargers would require a multi-objective or weighted cost function $w_t \cdot \text{minutes} + w_p \cdot \text{rupees}$.

### 3.2 Algorithms Left Out on Purpose
Two families of algorithms from the course syllabus were deliberately excluded because their problem structure does not match single-agent route optimization or facility location:

1. **Adversarial Search (Minimax and Alpha–Beta Pruning)**:
   - Minimax and Alpha–Beta Pruning are designed for two-player, zero-sum turn-taking games (such as Tic-Tac-Toe, Chess, or Connect Four) where an opponent actively maximizes the agent's cost or minimizes its utility. Planning an EV trip across a road network is a single-agent deterministic optimization problem with no competing adversary, so adversarial game trees do not apply.
2. **Constraint Satisfaction Problems (Backtracking Search and Arc Consistency / AC-3)**:
   - CSP algorithms assign values from finite domains to a fixed set of variables subject to hard boolean constraints (e.g., Sudoku, map coloring, or timetable scheduling). Finding a minimum-time EV trajectory with variable-length paths and repeated charging actions is a sequential shortest-path optimization problem over a weighted state-transition graph rather than a fixed-variable constraint satisfaction problem.
