"""Informed and optimal cost search: Uniform Cost Search, Greedy Best-First, and A*.

Author: ATK

All solvers follow the standard project signature:
    solve(problem, trace=False, max_expansions=2_000_000, time_limit_s=30) -> Result

Evaluation functions f(s):
  - UCS:    f(s) = g(s)            (path cost in minutes from start)
  - Greedy: f(s) = h(s)            (estimated minutes remaining to destination)
  - A*:     f(s) = g(s) + h(s)     (estimated total trip minutes through s)
"""

from __future__ import annotations

import heapq
from itertools import count

from ev.heuristics import HeuristicFn, make_problem_heuristics
from ev.metrics import DEFAULT_MAX_EXPANSIONS, DEFAULT_TIME_LIMIT_S, Result, SearchRun
from ev.model import Action, Problem, State

# Maps each reached state to (parent_state, action_taken) or None for start.
Parents = dict[State, tuple[State, Action] | None]


def _rebuild_path(parents: Parents, state: State) -> list[Action]:
    """Walk parent links back to start and return the sequence of actions."""
    path: list[Action] = []
    link = parents[state]
    while link is not None:
        state, action = link
        path.append(action)
        link = parents[state]
    path.reverse()
    return path


def ucs(
    problem: Problem,
    trace: bool = False,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
) -> Result:
    """Uniform Cost Search (Dijkstra's algorithm on the state space).

    Expands states in order of increasing path cost g(s) (total minutes).
    Goal test is performed at expansion (pop) time, not generation time,
    because the first path to generate the goal may use a slow road, whereas
    the first path to POP the goal is guaranteed to have minimum total minutes.

    Complete: yes (all action costs are integers >= 1 minute, state space finite).
    Optimal: yes, minimizes total trip minutes.
    """
    run = SearchRun("UCS", trace, max_expansions, time_limit_s)
    start = problem.initial_state()

    best_g: dict[State, int] = {start: 0}
    parents: Parents = {start: None}
    expanded: set[State] = set()

    tie = count()
    # Heap entries: (g_cost, tie_breaker, state, depth)
    frontier: list[tuple[int, int, State, int]] = [(0, next(tie), start, 0)]
    run.update_frontier(1)

    while frontier:
        g, _, state, depth = heapq.heappop(frontier)

        # Skip stale heap entries if a cheaper path to `state` was already found.
        if state in expanded:
            continue
        expanded.add(state)

        if problem.is_goal(state):
            return run.solved(problem, _rebuild_path(parents, state))
        if run.limit_reached():
            return run.cutoff()

        run.record_expansion(state, depth, len(frontier))
        options = problem.successors(state)
        run.record_generated(len(options))

        for action, next_state, step_minutes in options:
            if next_state in expanded:
                continue
            new_g = g + step_minutes
            if new_g < best_g.get(next_state, float("inf")):
                best_g[next_state] = new_g
                parents[next_state] = (state, action)
                heapq.heappush(frontier, (new_g, next(tie), next_state, depth + 1))

        run.update_frontier(len(frontier))

    return run.no_solution()


def greedy_best_first(
    problem: Problem,
    trace: bool = False,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    heuristic: str | HeuristicFn = "H2",
) -> Result:
    """Greedy Best-First Search: expands the state with smallest h(s).

    Looks only at the heuristic estimate to the goal and ignores the time
    already spent g(s). Fast when the heuristic points cleanly toward a charger
    and the destination, but NOT optimal—can take slower roads or unnecessary
    detours because past cost is ignored.

    Complete: yes on this finite state space (explored set prevents loops).
    Optimal: no.
    """
    if isinstance(heuristic, str):
        h_name = heuristic
        h_fn = make_problem_heuristics(problem)[h_name]
        algo_label = f"Greedy({h_name})" if h_name != "H2" else "Greedy"
    else:
        h_fn = heuristic
        algo_label = "Greedy"

    run = SearchRun(algo_label, trace, max_expansions, time_limit_s)
    start = problem.initial_state()

    parents: Parents = {}
    expanded: set[State] = set()

    tie = count()
    # Heap entries: (h_val, tie_breaker, state, parent_link, depth)
    frontier: list[tuple[float, int, State, tuple[State, Action] | None, int]] = [
        (h_fn(start), next(tie), start, None, 0)
    ]
    run.update_frontier(1)

    while frontier:
        _h, _, state, link, depth = heapq.heappop(frontier)
        if state in expanded:
            continue
        expanded.add(state)
        parents[state] = link

        if problem.is_goal(state):
            return run.solved(problem, _rebuild_path(parents, state))
        if run.limit_reached():
            return run.cutoff()

        run.record_expansion(state, depth, len(frontier))
        options = problem.successors(state)
        run.record_generated(len(options))

        for action, next_state, _step_minutes in options:
            if next_state not in expanded:
                heapq.heappush(
                    frontier,
                    (h_fn(next_state), next(tie), next_state, (state, action), depth + 1),
                )

        run.update_frontier(len(frontier))

    return run.no_solution()


def astar(
    problem: Problem,
    trace: bool = False,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    heuristic: str | HeuristicFn = "H2",
) -> Result:
    """A* Search: expands states in order of f(s) = g(s) + h(s).

    With an admissible and consistent heuristic (H0, H1, or H2), A* is both
    complete and optimal (guarantees minimum total trip minutes), and never
    needs to re-open an already expanded state.

    Tie-breaking: when two states have the same f(s), ties are broken in favor
    of smaller h(s) (equivalently larger g(s)), so the search prefers states
    closer to the destination instead of fanning out among equal-cost paths.
    """
    if isinstance(heuristic, str):
        h_name = heuristic
        h_fn = make_problem_heuristics(problem)[h_name]
        algo_label = f"A*({h_name})" if h_name != "H2" else "A*"
    else:
        h_fn = heuristic
        algo_label = "A*"

    run = SearchRun(algo_label, trace, max_expansions, time_limit_s)
    start = problem.initial_state()

    best_g: dict[State, int] = {start: 0}
    parents: Parents = {start: None}
    expanded: set[State] = set()

    tie = count()
    h0 = h_fn(start)
    # Heap entries: (f_val, h_val, tie_breaker, g_val, state, depth)
    frontier: list[tuple[float, float, int, int, State, int]] = [
        (h0, h0, next(tie), 0, start, 0)
    ]
    run.update_frontier(1)

    while frontier:
        _f, _h, _, g, state, depth = heapq.heappop(frontier)
        if state in expanded:
            continue
        expanded.add(state)

        if problem.is_goal(state):
            return run.solved(problem, _rebuild_path(parents, state))
        if run.limit_reached():
            return run.cutoff()

        run.record_expansion(state, depth, len(frontier))
        options = problem.successors(state)
        run.record_generated(len(options))

        for action, next_state, step_minutes in options:
            if next_state in expanded:
                continue
            new_g = g + step_minutes
            if new_g < best_g.get(next_state, float("inf")):
                best_g[next_state] = new_g
                parents[next_state] = (state, action)
                h_next = h_fn(next_state)
                heapq.heappush(
                    frontier,
                    (new_g + h_next, h_next, next(tie), new_g, next_state, depth + 1),
                )

        run.update_frontier(len(frontier))

    return run.no_solution()


# Convenience wrappers for testing individual heuristics in experiments/checker.
def astar_h0(
    problem: Problem,
    trace: bool = False,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
) -> Result:
    """A* using H0 (zero heuristic)."""
    return astar(problem, trace, max_expansions, time_limit_s, heuristic="H0")


def astar_h1(
    problem: Problem,
    trace: bool = False,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
) -> Result:
    """A* using H1 (straight-line distance / max speed)."""
    return astar(problem, trace, max_expansions, time_limit_s, heuristic="H1")


def astar_h2(
    problem: Problem,
    trace: bool = False,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
) -> Result:
    """A* using H2 (straight-line drive time + forced minimum charge time)."""
    return astar(problem, trace, max_expansions, time_limit_s, heuristic="H2")


# Standard dictionary exposed for main.py (matches ev/uninformed.py).
ALGORITHMS = {
    "UCS": ucs,
    "Greedy": greedy_best_first,
    "A*": astar,
}