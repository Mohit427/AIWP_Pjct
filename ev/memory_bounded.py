"""Memory-bounded informed search (IDA*, DFBB) and Random Walk local search baseline.

Author: Rohiit
Deliverables:
  - Iterative Deepening A* (IDA*)
  - Depth-First Branch and Bound (DFBB) — supports both unbounded (inf) and
    heuristic/greedy-seeded initial upper bounds
  - Random Walk baseline for charger placement optimization

All route-finding solvers follow the standard project signature:
    solve(problem, trace=False, max_expansions=2_000_000, time_limit_s=30) -> Result
"""

from __future__ import annotations

import math
import random
import time

from ev.charger_placement import (
    ChargerPlacementProblem,
    LocalSearchResult,
    evaluate_charger_set,
)
from ev.heuristics import HeuristicFn, make_problem_heuristics
from ev.informed import greedy_best_first
from ev.metrics import DEFAULT_MAX_EXPANSIONS, DEFAULT_TIME_LIMIT_S, Result, SearchRun
from ev.model import Action, Problem, State

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

    Performs series of depth-first searches bounded by an increasing f-cost
    threshold f(s) = g(s) + ceil(h(s)).
      - Complete: yes (positive integer step costs, finite state space).
      - Optimal: yes, when h(s) is admissible and consistent.
      - Memory: O(b * d) active frontier along the current path, plus per-pass
        cycle/transposition dominance pruning freed between iterations.
    """
    h_fn, algo_label = _resolve_heuristic(problem, heuristic, "IDA*")
    run = SearchRun(algo_label, trace, max_expansions, time_limit_s)
    start = problem.initial_state()

    if problem.is_goal(start):
        return run.solved(problem, [])

    threshold = _int_heuristic(h_fn, start)

    while True:
        if trace:
            print(f"--- {algo_label} threshold {threshold} min ---")

        next_threshold = float("inf")
        best_g: dict[State, int] = {start: 0}

        def ordered_children(
            state: State, g: int, on_path: set[State]
        ) -> list[tuple[int, int, Action, State]]:
            nonlocal next_threshold
            options = problem.successors(state)
            run.record_generated(len(options))
            valid: list[tuple[int, int, Action, State]] = []
            for action, nxt, step_minutes in options:
                if nxt in on_path:
                    continue
                ng = g + step_minutes
                nf = ng + _int_heuristic(h_fn, nxt)
                if nf > threshold:
                    if nf < next_threshold:
                        next_threshold = nf
                    continue
                if ng >= best_g.get(nxt, float("inf")):
                    continue
                best_g[nxt] = ng
                # Sort by smallest f(s), breaking ties toward larger g(s) (deeper progress)
                valid.append((nf, -ng, action, nxt))
            valid.sort()
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

            _nf, neg_ng, action, state = item
            ng = -neg_ng
            frontier_size -= 1

            # Skip if a cheaper path to `state` was discovered after this entry was queued
            if ng > best_g.get(state, float("inf")):
                continue

            if problem.is_goal(state):
                return run.solved(problem, path_actions + [action])

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

        if next_threshold == float("inf"):
            return run.no_solution()
        threshold = int(next_threshold)


def _greedy_probe_bound(
    problem: Problem,
    h_fn: HeuristicFn,
    max_steps: int = 500,
) -> tuple[int | None, list[Action] | None]:
    """Run a lightweight O(d)-memory greedy depth-first walk to seed DFBB's upper bound.

    Avoids allocating a full A*/Greedy priority queue so DFBB's memory footprint
    remains genuinely memory-bounded while still starting with a tight upper bound.
    Falls back to greedy_best_first if the single-branch probe hits a dead end.
    """
    state = problem.initial_state()
    visited: set[State] = {state}
    path: list[Action] = []
    total_minutes = 0

    for _ in range(max_steps):
        if problem.is_goal(state):
            return total_minutes, path
        candidates = [
            (h_fn(nxt), step_min, action, nxt)
            for action, nxt, step_min in problem.successors(state)
            if nxt not in visited
        ]
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1]))
        _h, step_min, action, nxt = candidates[0]
        visited.add(nxt)
        path.append(action)
        total_minutes += step_min
        state = nxt

    if problem.is_goal(state):
        return total_minutes, path

    # Fallback: quick bounded Greedy Best-First search
    res = greedy_best_first(
        problem,
        trace=False,
        max_expansions=min(1000, DEFAULT_MAX_EXPANSIONS),
        heuristic=h_fn,
    )
    if res.found and res.total_minutes is not None:
        return res.total_minutes, list(res.path)
    return None, None


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

    Explores the search space depth-first while maintaining the best solution
    cost `best_bound` found so far (seeded optionally by a fast greedy bound
    or a caller-supplied `initial_bound`). Any branch with f(s) = g(s) + h(s)
    >= best_bound (or > best_bound when seeking the first path at the exact
    initial bound) is pruned immediately.

      - Complete: yes on finite state space (provided initial_bound >= optimal cost).
      - Optimal: yes, when h(s) is admissible and initial_bound >= optimal cost.
      - Memory: O(b * d) active frontier stack.
    """
    h_fn, algo_label = _resolve_heuristic(problem, heuristic, "DFBB")
    run = SearchRun(algo_label, trace, max_expansions, time_limit_s)
    start = problem.initial_state()

    if problem.is_goal(start):
        return run.solved(problem, [])

    best_cost: float = float("inf") if initial_bound is None else float(initial_bound)
    best_path: list[Action] | None = None

    if initial_bound is None and use_greedy_bound:
        probe_cost, probe_path = _greedy_probe_bound(problem, h_fn)
        if probe_cost is not None and probe_path is not None:
            best_cost = float(probe_cost)
            best_path = probe_path
            if trace:
                print(f"--- {algo_label} seeded initial bound = {best_cost:.0f} min ---")

    best_g: dict[State, int] = {start: 0}

    def can_prune(nf: int) -> bool:
        """Prune if nf cannot improve on best_cost.

        If we already hold a concrete `best_path` of cost `best_cost`, we only
        need paths strictly cheaper (`nf >= best_cost` is pruned). If `initial_bound`
        was passed as a numeric target without a path yet, allow `nf == best_cost`
        until a path achieving it is recorded.
        """
        if best_path is not None:
            return nf >= best_cost
        return nf > best_cost

    def ordered_children(
        state: State, g: int, on_path: set[State]
    ) -> list[tuple[int, int, Action, State]]:
        options = problem.successors(state)
        run.record_generated(len(options))
        valid: list[tuple[int, int, Action, State]] = []
        for action, nxt, step_minutes in options:
            if nxt in on_path:
                continue
            ng = g + step_minutes
            nf = ng + _int_heuristic(h_fn, nxt)
            if can_prune(nf):
                continue
            if ng >= best_g.get(nxt, float("inf")):
                continue
            valid.append((nf, -ng, action, nxt))
        valid.sort()
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

        nf, neg_ng, action, state = item
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
            if trace:
                print(f"--- {algo_label} new best solution: {ng} min ---")
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
    """DFBB starting with bound = infinity (no greedy warm-start)."""
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


def random_walk(
    problem: ChargerPlacementProblem,
    initial_state: frozenset[str] | None = None,
    max_iterations: int = 100,
    trace: bool = False,
    seed: int | None = None,
) -> LocalSearchResult:
    """Random Walk baseline for charger placement optimization.

    At each step, picks a uniformly random neighbor (swapping one chosen charger
    with one unchosen candidate) and moves to it unconditionally regardless of
    whether it improves or worsens the average trip time, while tracking the
    best placement encountered across the walk.
    """
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


# Standard dictionary exposed for main.py and ATK's checker.py
ALGORITHMS = {
    "IDA*": ida_star,
    "DFBB": dfbb,
}