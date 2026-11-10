# Team contract

Everything the other modules need to plug in. Please do not change the model or
the `Result` fields without telling the team: all 12 algorithms depend on them.
Python 3.10+, standard library only (matplotlib is allowed in `ev/viz.py`).

## Who adds what

| File | Provides |
|---|---|
| `ev/informed.py` | an `ALGORITHMS` dictionary |
| `ev/local_search.py` | an `ALGORITHMS` dictionary |
| `ev/more_informed.py` | an `ALGORITHMS` dictionary |
| `ev/viz.py` | `visualize(problem, results)` |

`main.py` imports these only if they exist, so nothing else needs editing.
Once your module exposes the dictionary below, your algorithms appear in the
menu, the comparison table and the verdict automatically.

```python
ALGORITHMS = {"UCS": ucs, "A*": astar}
```

## Commands

```
python main.py                 # interactive menu
python main.py --demo          # non-interactive tour
python main.py --experiments   # regenerates results/
python -m unittest             # all tests
```

## The model (`ev/model.py`)

- **State**: `(node, battery)`, a tuple of `str` and `int`. Time is not in the
  state; it is the path cost.
- **Goal**: `node == destination`, at any battery level.
- **Action**: `Action(kind, target)`, where `kind` is `DRIVE` or `CHARGE`.

```python
problem.initial_state()      # -> (node, battery)
problem.is_goal(state)       # -> bool
problem.successors(state)    # -> [(action, next_state, step_cost_minutes), ...]
problem.replay(path)         # -> (final_state, total_minutes); ValueError if illegal
```

Successors are sorted by neighbour name with drives before charge. Pass
`shuffle_seed=<int>` when building a `Problem` for a repeatable random order.

Map data for heuristics and drawing, on `problem.road_map`:

```python
road_map.coords[node]                 # (x_km, y_km)
road_map.roads[node][neighbour]       # Road(km, minutes, battery_cost)
road_map.chargers[node]               # Charger(minutes_per_unit, price_per_unit)
road_map.straight_line_km(a, b)
```

Also available: `problem.start`, `problem.destination`, `problem.start_battery`,
`problem.battery_cap`.

Getting maps (`ev/maps.py`): `toy_problem()`, `corridor_problem()`, and
`generate_sized_map("small" | "medium" | "large").problem()`.

### Guarantees you can rely on

- Every road has `minutes >= 1` and an integer `battery_cost >= 1`.
- Every road's `km` is at least the straight-line distance between its ends.

The second guarantee gives an admissible time heuristic. A heuristic is
admissible if it never overestimates the remaining cost:

```python
max_km_per_minute = max(
    road.km / road.minutes
    for links in road_map.roads.values() for road in links.values()
)
h = road_map.straight_line_km(node, problem.destination) / max_km_per_minute
```

Compute `max_km_per_minute` once, not per call. Using an average speed instead
of the maximum can overestimate and lose A*'s optimality.

## Your solver

Exact signature:

```python
def solve(problem, trace=False, max_expansions=2_000_000, time_limit_s=30) -> Result:
```

Always use `SearchRun` from `ev/metrics.py`, so counting means the same for all:

```python
def my_search(problem, trace=False, max_expansions=2_000_000, time_limit_s=30):
    run = SearchRun("MyName", trace, max_expansions, time_limit_s)
    # In your main loop, in this order:
    if run.limit_reached():
        return run.cutoff()
    run.record_expansion(state, depth, len(frontier))   # after removing the state
    options = problem.successors(state)
    run.record_generated(len(options))
    # ... add to the frontier ...
    run.update_frontier(len(frontier))
    # To finish:
    return run.solved(problem, path)   # path is a list of Action
    return run.no_solution()           # the search proved there is no path
```

Rules:

1. Count **one expansion per call that asks for a state's successors**.
2. A limit that stops you is `run.cutoff()`. Only return `no_solution` when the
   search really proved no path exists. Local search that gives up without
   proof is therefore a `cutoff`.
3. `run.solved()` replays your path and raises an error if it is illegal or
   misses the goal. Fix the path; do not catch the error.
4. Do not fill in `total_minutes`, `runtime_s` or `peak_memory_kb` yourself.
5. **Randomized algorithms must be seeded inside the solver** (for example
   `random.Random(42)`), not with the global `random` module. `measure()` runs
   each solver twice and both runs must do the same work.

## Result fields

`algorithm`, `found`, `status` (`"solved"`, `"no_solution"`, `"cutoff"`), `path`,
`total_minutes`, `num_actions`, `nodes_expanded`, `nodes_generated`,
`peak_frontier`, `runtime_s`, `peak_memory_kb`, `expanded_order`.

- `total_minutes` and `num_actions` are `None` when no path was found.
- `peak_memory_kb` is only set by `measure(solver, problem)`. It is `None` if you
  call a solver directly.
- `expanded_order` is a list of states in expansion order, capped at 5000. It is
  empty in the memory run, so read it from the `Result` that `measure` returns.

## Visualization

`visualize(problem, results)` receives the `Problem` and a dictionary of
algorithm name to `Result`. Import matplotlib inside `ev/viz.py` only, so that
the rest of the project still runs on a computer without it.

## Verdict

`main.py` picks the winner from `Result` values and only solved runs qualify.
No names are hard-coded, so new algorithms join automatically.
