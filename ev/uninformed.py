"""Uninformed search: BFS, DFS and IDDFS.

"Uninformed" means these algorithms use only the problem definition. They have
no estimate of how close a state is to the goal.

All three share the standard signature:
    solve(problem, trace=False, max_expansions=2_000_000, time_limit_s=30) -> Result

Notation: b = branching factor (moves per state), d = depth of the shallowest
goal, V = reachable states, E = legal moves between them.
"""

from __future__ import annotations

from collections import deque
from itertools import count

from ev.metrics import DEFAULT_MAX_EXPANSIONS, DEFAULT_TIME_LIMIT_S, Result, SearchRun
from ev.model import Action, Problem, State

# Links each state to the (previous state, action) that first reached it.
Parents = dict[State, "tuple[State, Action] | None"]


def _rebuild_path(parents: Parents, state: State) -> list[Action]:
    """Walk parent links back to the start and return the actions in order.

    Storing one link per state is much cheaper than storing a full copy of
    the path with every frontier entry.
    """
    path: list[Action] = []
    link = parents[state]
    while link is not None:
        state, action = link
        path.append(action)
        link = parents[state]
    path.reverse()
    return path


def bfs(
    problem: Problem,
    trace: bool = False,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
) -> Result:
    """Breadth-first search with a FIFO queue and an explored set.

    Minimizes the NUMBER OF ACTIONS, NOT the minutes. It treats every action
    as equally expensive, so a 10-minute charge and a 160-minute drive both
    count as one step. The fewest-actions plan is often not the fastest.

    Complete: yes, the state space is finite and nothing reachable is skipped.
    Optimal: for action count only.
    Time and memory: O(V + E) here; in tree terms O(b^d).

    A state is marked explored when first generated, not when expanded, so it
    enters the queue at most once. The goal test also happens at generation:
    anything still in the queue is at least as deep, so waiting gains nothing.
    """
    run = SearchRun("BFS", trace, max_expansions, time_limit_s)
    start = problem.initial_state()
    parents: Parents = {start: None}  # its keys are the explored set
    if problem.is_goal(start):
        return run.solved(problem, [])

    frontier = deque([(start, 0)])
    run.update_frontier(1)
    while frontier:
        if run.limit_reached():
            return run.cutoff()
        state, depth = frontier.popleft()
        run.record_expansion(state, depth, len(frontier))
        for action, next_state, _minutes in problem.successors(state):
            run.record_generated()
            if next_state in parents:
                continue
            parents[next_state] = (state, action)
            if problem.is_goal(next_state):
                return run.solved(problem, _rebuild_path(parents, next_state))
            frontier.append((next_state, depth + 1))
        run.update_frontier(len(frontier))
    return run.no_solution()


def dfs(
    problem: Problem,
    trace: bool = False,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
) -> Result:
    """Depth-first search with an explicit stack and an explored set.

    Complete: yes on this finite space, because the explored set stops it
    from looping and every reachable state is eventually expanded.
    Optimal: no. It returns the first path it stumbles on.
    Time: O(V + E). Memory: O(V + E), the explored set plus a stack that can
    hold duplicates.

    An explicit stack is used so that long paths cannot hit Python's recursion
    limit. A state is marked explored when popped, not when pushed; marking
    on push would commit a state to an earlier parent and change the order
    away from true depth-first. The stack may therefore hold duplicates, and
    stale copies are skipped when popped.
    """
    run = SearchRun("DFS", trace, max_expansions, time_limit_s)
    start = problem.initial_state()
    parents: Parents = {}  # its keys are the explored set
    stack: list[tuple[State, tuple[State, Action] | None, int]] = [(start, None, 0)]
    run.update_frontier(1)

    while stack:
        state, link, depth = stack.pop()
        if state in parents:
            continue
        parents[state] = link
        if problem.is_goal(state):
            return run.solved(problem, _rebuild_path(parents, state))
        if run.limit_reached():
            return run.cutoff()
        run.record_expansion(state, depth, len(stack))
        options = problem.successors(state)
        run.record_generated(len(options))
        # Reversed, so the first option in successor order is popped first.
        for action, next_state, _minutes in reversed(options):
            if next_state not in parents:
                stack.append((next_state, (state, action), depth + 1))
        run.update_frontier(len(stack))
    return run.no_solution()


# Outcomes of one depth-limited pass.
_FOUND = "found"
_GO_DEEPER = "go_deeper"  # the depth limit pruned something: try a larger limit
_EXHAUSTED = "exhausted"  # nothing was pruned, so a larger limit cannot help
_STOPPED = "stopped"  # max_expansions or the time limit was hit


def _depth_limited(
    problem: Problem, limit: int, run: SearchRun
) -> tuple[str, list[Action] | None]:
    """Run one depth-first pass that goes no deeper than `limit` actions.

    Keeps only the current path and each level's untried options, which is
    where the low memory of IDDFS comes from.

    Uses an ancestor check only: a state is skipped if it is already on the
    CURRENT path. A global explored set must not be used here. A state first
    reached near the limit would be marked explored, and a later shorter route
    to it would be rejected, even though only that route has enough remaining
    depth to reach the goal. That would make the search miss solutions.
    """
    start = problem.initial_state()
    if problem.is_goal(start):
        return _FOUND, []
    if limit == 0:
        return _GO_DEEPER, None
    if run.limit_reached():
        return _STOPPED, None

    def open_state(state: State, on_path: set[State]) -> list[tuple[Action, State]]:
        """Fetch a state's successors, dropping those that would form a cycle."""
        options = problem.successors(state)
        run.record_generated(len(options))
        return [(action, nxt) for action, nxt, _minutes in options if nxt not in on_path]

    run.record_expansion(start, 0, 0)
    on_path = {start}
    current_states = [start]
    path: list[Action] = []
    children = open_state(start, on_path)
    untried = [iter(children)]  # one iterator per level of the current path
    frontier_size = len(children)
    run.update_frontier(frontier_size)
    pruned_by_depth = False

    while untried:
        child = next(untried[-1], None)
        if child is None:  # this level is finished: step back up
            untried.pop()
            on_path.discard(current_states.pop())
            if path:
                path.pop()
            continue

        action, state = child
        frontier_size -= 1
        if problem.is_goal(state):
            return _FOUND, path + [action]
        depth = len(path) + 1
        if depth == limit:
            pruned_by_depth = True
            continue
        if run.limit_reached():
            return _STOPPED, None

        run.record_expansion(state, depth, frontier_size)
        children = open_state(state, on_path)
        frontier_size += len(children)
        run.update_frontier(frontier_size)
        path.append(action)
        current_states.append(state)
        on_path.add(state)
        untried.append(iter(children))

    return (_GO_DEEPER if pruned_by_depth else _EXHAUSTED), None


def iddfs(
    problem: Problem,
    trace: bool = False,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
) -> Result:
    """Iterative deepening: depth-limited DFS with limits 0, 1, 2, ...

    Complete: yes. Every shallower limit fails first, so the first goal found
    is at the minimum depth.
    Optimal: for action count only, the same as BFS.
    Time: O(b^d). Memory: O(b*d), only the current path and its siblings.

    Repeating the shallow levels is cheap in a tree that grows quickly,
    because most nodes are in the deepest level. The counters add up work
    across all passes, so the repetition is reported and not hidden.

    Unlike BFS it has no explored set, so it can reach the same state along
    many different paths. Where many paths lead to the same states, as with
    charging, that can make it far slower than BFS.

    If a whole pass finishes without the depth limit pruning anything, every
    cycle-free path has been tried, so the result is proven no_solution.
    """
    run = SearchRun("IDDFS", trace, max_expansions, time_limit_s)
    for limit in count():
        if trace:
            print(f"--- depth limit {limit} ---")
        outcome, path = _depth_limited(problem, limit, run)
        if outcome == _FOUND:
            return run.solved(problem, path)
        if outcome == _EXHAUSTED:
            return run.no_solution()
        if outcome == _STOPPED:
            return run.cutoff()
    raise AssertionError("unreachable: count() never ends")


# Used by main.py. Teammates' modules expose the same kind of dictionary.
ALGORITHMS = {"BFS": bfs, "DFS": dfs, "IDDFS": iddfs}
