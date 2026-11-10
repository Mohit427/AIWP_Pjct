"""Memory-bounded informed search (IDA*, DFBB) and Random Walk baseline.

Implements:
  - Iterative Deepening A* (IDA*) with integer f-contours and admissible
    Bellman subtree bound learning
  - Depth-First Branch and Bound (DFBB), testable with and without an initial
    upper bound
  - Random Walk baseline supporting both route-finding (`Problem` -> `Result`)
    and charger placement (`ChargerPlacementProblem` -> `LocalSearchResult`)

All route-finding solvers follow the CONTRACT.md signature:
    solve(problem, trace=False, max_expansions=2_000_000, time_limit_s=30) -> Result
"""

from __future__ import annotations

import csv
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from ev.charger_placement import (
    ChargerPlacementProblem,
    LocalSearchResult,
    evaluate_charger_set,
)
from ev.heuristics import HeuristicFn, make_problem_heuristics
from ev.maps import (
    DEFAULT_SEEDS,
    MAP_SIZES,
    corridor_problem,
    generate_map,
    generate_sized_map,
    seeded_start_goal_pairs,
    toy_problem,
)
from ev.metrics import (
    CUTOFF,
    DEFAULT_MAX_EXPANSIONS,
    DEFAULT_TIME_LIMIT_S,
    NO_SOLUTION,
    SOLVED,
    Result,
    SearchRun,
    measure,
)
from ev.model import Action, MAX_BATTERY_CAP, Problem, State

EPSILON = 1e-9


def _resolve_heuristic(
    problem: Problem,
    heuristic: str | HeuristicFn,
    base_name: str,
) -> tuple[HeuristicFn, str]:
    """Return (heuristic_fn, algorithm_label) for a solver run."""
    if isinstance(heuristic, str):
        h_name = heuristic
        h_fn = make_problem_heuristics(problem)[h_name]
        label = f"{base_name}({h_name})" if h_name != "H2" else base_name
    else:
        h_fn = heuristic
        label = base_name
    return h_fn, label


def _int_heuristic(h_fn: HeuristicFn, state: State) -> int:
    """Return ceil(h(state)) as a non-negative integer.

    Because every action cost in the model (road minutes and charger minutes
    per unit) is a positive integer, the true optimal remaining cost h*(s) is
    always an integer. For any admissible heuristic h(s) <= h*(s),
    ceil(h(s)) <= ceil(h*(s)) = h*(s) remains admissible and consistent while
    keeping f-thresholds on integer minute boundaries.
    """
    val = h_fn(state)
    if val <= 0.0:
        return 0
    return math.ceil(val - EPSILON)


def ida_star(
    problem: Problem,
    trace: bool = False,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    heuristic: str | HeuristicFn = "H2",
) -> Result:
    """Iterative Deepening A* (IDA*).

    Performs depth-first contours bounded by increasing integer f-cost
    thresholds f(s) = g(s) + ceil(h(s)), with admissible Bellman subtree bound
    backups across passes so cyclic road+charging networks do not cause
    unit-step threshold thrashing.
      - Complete: yes (positive integer step costs, finite state space).
      - Optimal: yes, when h(s) is admissible and consistent.
      - Memory: O(b * d) active frontier along the current path.
    """
    h_fn, algo_label = _resolve_heuristic(problem, heuristic, "IDA*")
    run = SearchRun(algo_label, trace, max_expansions, time_limit_s)
    start = problem.initial_state()

    if problem.is_goal(start):
        return run.solved(problem, [])

    def h_ceil(state: State) -> int:
        return _int_heuristic(h_fn, state)

    learned_h: dict[State, int | float] = {}

    def eff_h(state: State) -> int | float:
        base = h_ceil(state)
        lh = learned_h.get(state, 0)
        return base if base >= lh else lh

    threshold: int | float = eff_h(start)
    max_seen_states = 0
    stagnant_passes = 0

    while True:
        next_threshold = float("inf")
        best_g: dict[State, int] = {start: 0}
        raw_pruned = False

        if run.limit_reached():
            return run.cutoff()

        def expand_node(
            state: State, g: int, on_path: set[State]
        ) -> tuple[list[tuple[int | float, int, Action, State]], float]:
            nonlocal next_threshold, raw_pruned
            options = problem.successors(state)
            run.record_generated(len(options))
            children: list[tuple[int | float, int, Action, State]] = []
            local_min_f = float("inf")
            for action, nxt, step_min in options:
                ng = g + step_min
                nf = ng + eff_h(nxt)
                if nxt in on_path or nxt == state:
                    if nf < local_min_f:
                        local_min_f = nf
                    continue
                if ng >= best_g.get(nxt, float("inf")):
                    if nf < local_min_f:
                        local_min_f = nf
                    continue
                if nf > threshold:
                    if nxt not in learned_h:
                        raw_pruned = True
                    if nf < local_min_f:
                        local_min_f = nf
                    if nf < next_threshold:
                        next_threshold = nf
                    continue
                best_g[nxt] = ng
                children.append((nf, -ng, action, nxt))
            children.sort()
            return children, local_min_f

        run.record_expansion(start, 0, 0)
        init_children, init_min_f = expand_node(start, 0, {start})
        frontier_size = len(init_children)
        run.update_frontier(frontier_size)
        # Stack frame: [state, g, depth, children, child_idx, min_sub_f, action_taken]
        stack: list[list] = [[start, 0, 0, init_children, 0, init_min_f, None]]
        on_path: set[State] = {start}

        while stack:
            frame = stack[-1]
            state, g, depth, children, idx, min_sub_f, _ = frame
            if idx >= len(children):
                if min_sub_f > threshold:
                    new_h = (
                        float("inf")
                        if min_sub_f == float("inf")
                        else int(min_sub_f) - g
                    )
                    if new_h > learned_h.get(state, 0):
                        learned_h[state] = new_h
                stack.pop()
                on_path.discard(state)
                if stack and min_sub_f < stack[-1][5]:
                    stack[-1][5] = min_sub_f
                continue

            _nf, neg_ng, action, nxt = children[idx]
            frame[4] = idx + 1
            ng = -neg_ng
            frontier_size -= 1

            if ng > best_g.get(nxt, float("inf")):
                cur_nf = ng + eff_h(nxt)
                if cur_nf < frame[5]:
                    frame[5] = cur_nf
                continue

            cur_nf = ng + eff_h(nxt)
            if cur_nf > threshold:
                if cur_nf < frame[5]:
                    frame[5] = cur_nf
                if cur_nf < next_threshold:
                    next_threshold = cur_nf
                continue

            if problem.is_goal(nxt):
                path = [f[6] for f in stack[1:]] + [action]
                return run.solved(problem, path)

            if run.limit_reached():
                return run.cutoff()

            nxt_depth = depth + 1
            run.record_expansion(nxt, nxt_depth, frontier_size)
            nxt_children, nxt_min_f = expand_node(nxt, ng, on_path | {nxt})
            frontier_size += len(nxt_children)
            run.update_frontier(frontier_size)
            on_path.add(nxt)
            stack.append([nxt, ng, nxt_depth, nxt_children, 0, nxt_min_f, action])

        if next_threshold == float("inf") or eff_h(start) == float("inf"):
            return run.no_solution()
        if not raw_pruned and len(learned_h) <= max_seen_states:
            stagnant_passes += 1
            if stagnant_passes >= len(learned_h) + 2:
                return run.no_solution()
        else:
            stagnant_passes = 0
            if len(learned_h) > max_seen_states:
                max_seen_states = len(learned_h)
        threshold = max(int(next_threshold), int(eff_h(start)))


def dfbb(
    problem: Problem,
    trace: bool = False,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    heuristic: str | HeuristicFn = "H2",
    initial_bound: float | None = None,
    use_greedy_bound: bool = True,
) -> Result:
    """Depth-First Branch and Bound (DFBB).

    Explores the state space depth-first while maintaining the best solution
    cost `best_cost` found so far. Any branch with f(s) = g(s) + ceil(h(s)) >=
    best_cost is pruned immediately.

      - `initial_bound`: optional caller-supplied starting upper bound (minutes).
      - `use_greedy_bound`: when True (default) and no solution has been found
        yet, orders children by smallest h(s) first so the initial DFS plunge
        quickly reaches the destination to establish a tight upper bound, then
        switches to smallest f(s) ordering. When False (`dfbb_unbounded`), uses
        plain f(s)/g(s) ordering with bound = infinity.
    """
    h_fn, algo_label = _resolve_heuristic(problem, heuristic, "DFBB")
    run = SearchRun(algo_label, trace, max_expansions, time_limit_s)
    start = problem.initial_state()

    if problem.is_goal(start):
        return run.solved(problem, [])

    def h_ceil(state: State) -> int:
        return _int_heuristic(h_fn, state)

    best_cost: float = float("inf") if initial_bound is None else float(initial_bound)
    best_path: list[Action] | None = None
    best_g: dict[State, int] = {start: 0}

    def can_prune(nf: int) -> bool:
        if best_path is not None:
            return nf >= best_cost
        return nf > best_cost

    def ordered_children(
        state: State, g: int, on_path: set[State]
    ) -> list[tuple[int, int, int, Action, State]]:
        options = problem.successors(state)
        run.record_generated(len(options))
        valid: list[tuple[int, int, int, Action, State]] = []
        for action, nxt, step_minutes in options:
            if nxt in on_path or nxt == state:
                continue
            ng = g + step_minutes
            if ng >= best_g.get(nxt, float("inf")):
                continue
            nh = h_ceil(nxt)
            nf = ng + nh
            if can_prune(nf):
                continue
            valid.append((nf, nh, -ng, action, nxt))
        if use_greedy_bound and best_path is None and initial_bound is None:
            valid.sort(key=lambda item: (item[1], item[0], item[2]))
        else:
            valid.sort(key=lambda item: (item[0], -item[2], item[1]))
        return valid

    if run.limit_reached():
        return run.cutoff()

    run.record_expansion(start, 0, 0)
    children = ordered_children(start, 0, {start})
    untried = [iter(children)]
    path_states: list[State] = [start]
    path_actions: list[Action] = []
    on_path: set[State] = {start}
    frontier_size = len(children)
    run.update_frontier(frontier_size)

    while untried:
        item = next(untried[-1], None)
        if item is None:
            untried.pop()
            on_path.discard(path_states.pop())
            if path_actions:
                path_actions.pop()
            continue

        nf, _nh, neg_ng, action, state = item
        ng = -neg_ng
        frontier_size -= 1

        if can_prune(nf):
            continue
        if ng >= best_g.get(state, float("inf")):
            continue
        best_g[state] = ng

        if problem.is_goal(state):
            best_cost = float(ng)
            best_path = path_actions + [action]
            continue

        if run.limit_reached():
            return run.cutoff()

        depth = len(path_actions) + 1
        run.record_expansion(state, depth, frontier_size)
        children = ordered_children(state, ng, on_path | {state})
        frontier_size += len(children)
        run.update_frontier(frontier_size)
        path_actions.append(action)
        path_states.append(state)
        on_path.add(state)
        untried.append(iter(children))

    if best_path is not None:
        return run.solved(problem, best_path)
    return run.no_solution()


def dfbb_unbounded(
    problem: Problem,
    trace: bool = False,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    heuristic: str | HeuristicFn = "H2",
) -> Result:
    """DFBB without greedy plunge ordering or initial bound (for baseline comparison)."""
    res = dfbb(
        problem,
        trace=trace,
        max_expansions=max_expansions,
        time_limit_s=time_limit_s,
        heuristic=heuristic,
        initial_bound=None,
        use_greedy_bound=False,
    )
    res.algorithm = "DFBB(unbounded)"
    return res


def _random_walk_route(
    problem: Problem,
    trace: bool = False,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    seed: int = 42,
    max_steps: int = 200,
) -> Result:
    """Seeded random walk on a route-finding `Problem` (follows CONTRACT.md rules).

    Seeds its own `random.Random(seed)` instance inside the solver so `measure()`
    performs identical timed and memory runs. Because a random walk that hits its
    step limit or a dead end after leaving start has not proven no solution
    exists, it returns `run.cutoff()` unless `start` itself has zero legal moves.
    """
    run = SearchRun("Random Walk", trace, max_expansions, time_limit_s)
    rng = random.Random(seed)
    state = problem.initial_state()
    if problem.is_goal(state):
        return run.solved(problem, [])

    path: list[Action] = []
    step_limit = min(max_expansions, max_steps)

    for depth in range(step_limit):
        if run.limit_reached():
            return run.cutoff()
        run.record_expansion(state, depth, 1 if depth > 0 else 0)
        options = problem.successors(state)
        run.record_generated(len(options))
        if not options:
            if depth == 0:
                return run.no_solution()
            return run.cutoff()
        run.update_frontier(len(options))
        action, next_state, _minutes = rng.choice(options)
        path.append(action)
        state = next_state
        if problem.is_goal(state):
            return run.solved(problem, path)

    return run.cutoff()


def _random_walk_placement(
    problem: ChargerPlacementProblem,
    initial_state: frozenset[str] | None = None,
    max_iterations: int = 100,
    trace: bool = False,
    seed: int | None = None,
) -> LocalSearchResult:
    """Random Walk baseline for charger placement optimization (`ChargerPlacementProblem`)."""
    rng = random.Random(seed)
    start_time = time.perf_counter()

    current_state = (
        problem.initial_state(seed) if initial_state is None else initial_state
    )
    current_value = evaluate_charger_set(problem, current_state)
    best_state = current_state
    best_value = current_value
    trace_history: list[dict] = []

    if trace:
        trace_history.append(
            {
                "iteration": 0,
                "current_value": current_value,
                "current_state": set(current_state),
                "best_value": best_value,
                "phase": "initial",
            }
        )

    iterations_done = 0
    for iteration in range(1, max_iterations + 1):
        unchosen = [c for c in problem.candidates if c not in current_state]
        if not unchosen or not current_state:
            break
        iterations_done = iteration
        neighbor_state = problem.random_neighbor(current_state, rng)
        neighbor_value = evaluate_charger_set(problem, neighbor_state)

        current_state = neighbor_state
        current_value = neighbor_value
        improved = False
        if current_value < best_value:
            best_value = current_value
            best_state = current_state
            improved = True

        if trace:
            trace_history.append(
                {
                    "iteration": iteration,
                    "current_value": current_value,
                    "current_state": set(current_state),
                    "neighbor_value": neighbor_value,
                    "neighbor_state": set(neighbor_state),
                    "accepted": True,
                    "best_value": best_value,
                    "improvement": improved,
                    "phase": "step",
                }
            )

    runtime = time.perf_counter() - start_time
    return LocalSearchResult(
        algorithm="Random Walk",
        best_state=best_state,
        best_value=best_value,
        iterations=iterations_done,
        runtime_s=runtime,
        trace_history=trace_history,
    )


def random_walk(
    problem: Problem | ChargerPlacementProblem,
    initial_state: frozenset[str] | None = None,
    max_iterations: int = 100,
    trace: bool = False,
    seed: int | None = None,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
) -> Result | LocalSearchResult:
    """Unified Random Walk supporting both ChargerPlacementProblem and route Problem."""
    if isinstance(problem, ChargerPlacementProblem):
        return _random_walk_placement(
            problem,
            initial_state=initial_state,
            max_iterations=max_iterations,
            trace=trace,
            seed=seed,
        )
    return _random_walk_route(
        problem,
        trace=trace,
        max_expansions=max_expansions,
        time_limit_s=time_limit_s,
        seed=42 if seed is None else seed,
        max_steps=max_iterations,
    )


# Standard dictionary exposed for main.py (via ev.more_informed) and ATK's checker.py
ALGORITHMS = {
    "IDA*": ida_star,
    "DFBB": dfbb,
}


# ---------------------------------------------------------------------------
# Wednesday (11 Nov) Experiment Runner for IDA* and DFBB across Map Sizes
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "map_size",
    "num_nodes",
    "map_seed",
    "battery_cap",
    "start_battery",
    "pair_index",
    "start",
    "goal",
    "algorithm",
    "status",
    "num_actions",
    "total_minutes",
    "nodes_expanded",
    "nodes_generated",
    "peak_frontier",
    "runtime_s",
    "peak_memory_kb",
]


def run_memory_bounded_experiments(
    results_dir: Path | str = Path(__file__).resolve().parent.parent / "results",
    caps: tuple[int, ...] = (4, 6, 10),
    num_pairs: int = 10,
    pair_seed: int = 7,
) -> list[dict]:
    """Run IDA*, DFBB, and DFBB(unbounded) across map sizes and caps, saving CSV and summary."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    csv_path = results_path / "memory_bounded_experiments.csv"

    solvers = {
        "IDA*": ida_star,
        "DFBB": dfbb,
        "DFBB(unbounded)": dfbb_unbounded,
    }

    rows: list[dict] = []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        # First record the standard toy and corridor benchmarks
        for std_name, std_prob in (("toy", toy_problem()), ("corridor", corridor_problem())):
            for algo_name, solver in solvers.items():
                res = measure(solver, std_prob)
                row = {
                    "map_size": std_name,
                    "num_nodes": std_prob.road_map.num_nodes(),
                    "map_seed": 0,
                    "battery_cap": std_prob.battery_cap,
                    "start_battery": std_prob.start_battery,
                    "pair_index": 0,
                    "start": std_prob.start,
                    "goal": std_prob.destination,
                    "algorithm": algo_name,
                    "status": res.status,
                    "num_actions": res.num_actions,
                    "total_minutes": res.total_minutes,
                    "nodes_expanded": res.nodes_expanded,
                    "nodes_generated": res.nodes_generated,
                    "peak_frontier": res.peak_frontier,
                    "runtime_s": res.runtime_s,
                    "peak_memory_kb": res.peak_memory_kb,
                }
                writer.writerow(row)
                handle.flush()
                rows.append(row)

        # Then run the 90 seeded trips across small, medium, large and battery caps 4, 6, 10
        for size, num_nodes in MAP_SIZES.items():
            generated = generate_map(num_nodes, DEFAULT_SEEDS[size], MAX_BATTERY_CAP)
            pairs = seeded_start_goal_pairs(
                generated.road_map,
                num_pairs,
                pair_seed,
                MAX_BATTERY_CAP,
                MAX_BATTERY_CAP // 2,
            )
            for cap in caps:
                for index, (start, goal) in enumerate(pairs):
                    problem = Problem(generated.road_map, start, goal, cap // 2, cap)
                    for algo_name, solver in solvers.items():
                        res = measure(solver, problem)
                        row = {
                            "map_size": size,
                            "num_nodes": num_nodes,
                            "map_seed": generated.seed_used,
                            "battery_cap": cap,
                            "start_battery": cap // 2,
                            "pair_index": index,
                            "start": start,
                            "goal": goal,
                            "algorithm": algo_name,
                            "status": res.status,
                            "num_actions": res.num_actions,
                            "total_minutes": res.total_minutes,
                            "nodes_expanded": res.nodes_expanded,
                            "nodes_generated": res.nodes_generated,
                            "peak_frontier": res.peak_frontier,
                            "runtime_s": res.runtime_s,
                            "peak_memory_kb": res.peak_memory_kb,
                        }
                        writer.writerow(row)
                        handle.flush()
                        rows.append(row)

    summary_text = summarize_memory_bounded(rows)
    (results_path / "memory_bounded_summary.txt").write_text(summary_text + "\n")
    return rows


def summarize_memory_bounded(rows: list[dict]) -> str:
    """Format a summary table for IDA*, DFBB, and DFBB(unbounded) runs."""
    by_group: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_group[(r["map_size"], int(r["battery_cap"]), r["algorithm"])].append(r)

    headers = [
        "map",
        "cap",
        "algorithm",
        "solved",
        "no_sol",
        "cutoff",
        "minutes",
        "expanded",
        "frontier",
        "time(ms)",
        "mem(KB)",
    ]
    table_rows: list[list[str]] = []
    for (map_size, cap, algo), grp in sorted(
        by_group.items(),
        key=lambda x: (
            {"toy": 0, "corridor": 1, "small": 2, "medium": 3, "large": 4}.get(
                x[0][0], 9
            ),
            x[0][1],
            x[0][2],
        ),
    ):
        solved_rows = [d for d in grp if d["status"] == SOLVED]
        n_solved = len(solved_rows)
        n_nosol = sum(1 for d in grp if d["status"] == NO_SOLUTION)
        n_cut = sum(1 for d in grp if d["status"] == CUTOFF)
        mean_min = (
            f"{statistics.fmean(d['total_minutes'] for d in solved_rows):.1f}"
            if solved_rows
            else "-"
        )
        mean_exp = (
            f"{statistics.fmean(d['nodes_expanded'] for d in solved_rows):.1f}"
            if solved_rows
            else "-"
        )
        mean_front = (
            f"{statistics.fmean(d['peak_frontier'] for d in solved_rows):.1f}"
            if solved_rows
            else "-"
        )
        mean_ms = (
            f"{statistics.fmean(d['runtime_s'] * 1000 for d in solved_rows):.2f}"
            if solved_rows
            else "-"
        )
        mean_mem = (
            f"{statistics.fmean(d['peak_memory_kb'] for d in solved_rows):.1f}"
            if solved_rows
            else "-"
        )
        table_rows.append(
            [
                map_size,
                str(cap),
                algo,
                str(n_solved),
                str(n_nosol),
                str(n_cut),
                mean_min,
                mean_exp,
                mean_front,
                mean_ms,
                mean_mem,
            ]
        )

    widths = [
        max(len(row[i]) for row in [headers] + table_rows) for i in range(len(headers))
    ]
    lines = [
        "MEMORY-BOUNDED SEARCH EXPERIMENT SUMMARY (IDA*, DFBB, DFBB-unbounded)",
        "",
        "  ".join(h.rjust(w) for h, w in zip(headers, widths)),
        "  ".join("-" * w for w in widths),
    ]
    for row in table_rows:
        lines.append("  ".join(c.rjust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)


if __name__ == "__main__":
    exp_rows = run_memory_bounded_experiments()
    print(summarize_memory_bounded(exp_rows))