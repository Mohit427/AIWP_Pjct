"""The standard Result record and the measurement harness.

Every algorithm uses SearchRun for its bookkeeping. Comparing algorithms is
only fair if "node expanded" and "cutoff" mean the same thing in all of them,
so that logic lives here once instead of being rewritten twelve times.

Vocabulary
----------
generated  a successor state was created.
expanded   a state was taken from the frontier and its successors were asked for.
frontier   states generated but not yet expanded (the queue or stack).
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable

from ev.model import Action, Problem, State

SOLVED = "solved"
NO_SOLUTION = "no_solution"
CUTOFF = "cutoff"

EXPANDED_ORDER_CAP = 5000
DEFAULT_MAX_EXPANSIONS = 2_000_000
DEFAULT_TIME_LIMIT_S = 30

# True in normal runs. measure() turns it off during its memory run, because
# expanded_order is a log kept for visualization and not part of any algorithm.
# Keeping up to 5000 states alive would add a similar amount to every result
# and hide the real difference between a low-memory algorithm and the others.
_record_expanded_order = True


@dataclass
class Result:
    """The outcome of one search. Every algorithm returns exactly this.

    total_minutes and num_actions are None when no path was found, because 0
    would be indistinguishable from a real zero-length trip (start == goal).
    """

    algorithm: str
    found: bool
    status: str  # SOLVED, NO_SOLUTION or CUTOFF
    path: list[Action] = field(default_factory=list)
    total_minutes: int | None = None
    num_actions: int | None = None
    nodes_expanded: int = 0
    nodes_generated: int = 0
    peak_frontier: int = 0
    runtime_s: float = 0.0
    peak_memory_kb: float | None = None  # filled in by measure()
    expanded_order: list[State] = field(default_factory=list)


class SearchRun:
    """Counters, limits and trace output for one execution of one algorithm."""

    def __init__(
        self,
        algorithm: str,
        trace: bool = False,
        max_expansions: int = DEFAULT_MAX_EXPANSIONS,
        time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    ) -> None:
        """Start the clock and zero the counters."""
        self.algorithm = algorithm
        self.trace = trace
        self.max_expansions = max_expansions
        self.time_limit_s = time_limit_s
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.peak_frontier = 0
        self.expanded_order: list[State] = []
        self._started = time.perf_counter()

    def limit_reached(self) -> bool:
        """Return True if the search must stop now. Call before each expansion."""
        return (
            self.nodes_expanded >= self.max_expansions
            or time.perf_counter() - self._started >= self.time_limit_s
        )

    def record_expansion(self, state: State, depth: int, frontier_size: int) -> None:
        """Log one expansion. frontier_size is measured after removing this state."""
        self.nodes_expanded += 1
        self.update_frontier(frontier_size)
        if _record_expanded_order and len(self.expanded_order) < EXPANDED_ORDER_CAP:
            self.expanded_order.append(state)
        if self.trace:
            print(
                f"step {self.nodes_expanded:>6} | state {state!s:<26} "
                f"| depth {depth:>4} | frontier {frontier_size:>6}"
            )

    def record_generated(self, count: int = 1) -> None:
        """Count newly created successor states."""
        self.nodes_generated += count

    def update_frontier(self, size: int) -> None:
        """Remember the largest frontier seen. Call after adding successors."""
        if size > self.peak_frontier:
            self.peak_frontier = size

    def _result(self, found: bool, status: str) -> Result:
        """Stop the clock and package the counters."""
        return Result(
            algorithm=self.algorithm,
            found=found,
            status=status,
            nodes_expanded=self.nodes_expanded,
            nodes_generated=self.nodes_generated,
            peak_frontier=self.peak_frontier,
            runtime_s=time.perf_counter() - self._started,
            expanded_order=self.expanded_order,
        )

    def solved(self, problem: Problem, path: list[Action]) -> Result:
        """Return a solved Result after independently checking the path.

        The minutes come from replaying the path through the model, not from
        the algorithm's own running total. An illegal path raises ValueError,
        and one that stops short of the goal raises RuntimeError: a buggy
        algorithm should fail loudly and never report a wrong answer.
        """
        result = self._result(True, SOLVED)
        final_state, minutes = problem.replay(path)
        if not problem.is_goal(final_state):
            raise RuntimeError(f"{self.algorithm}: path ends at {final_state}, not the goal")
        result.path = list(path)
        result.total_minutes = minutes
        result.num_actions = len(path)
        return result

    def no_solution(self) -> Result:
        """Return a Result meaning the search finished and no path exists."""
        return self._result(False, NO_SOLUTION)

    def cutoff(self) -> Result:
        """Return a Result meaning a limit stopped the search, so the answer is unknown."""
        return self._result(False, CUTOFF)


Solver = Callable[..., Result]


def measure(
    solver: Solver,
    problem: Problem,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
) -> Result:
    """Run a solver twice: once for time, once for memory. Return one Result.

    Two runs are needed because tracemalloc records every allocation, which
    slows the program and would corrupt the runtime figure.

    The memory run repeats the same work as the timed run. It is bounded by
    the timed run's expansion count instead of by the clock, so a slower run
    cannot be cut off earlier and report a smaller peak. One extra expansion is
    allowed so the limit cannot trigger just before a goal the timed run found.
    Algorithms that use randomness must be seeded for the two runs to match.
    """
    global _record_expanded_order
    timed = solver(
        problem, trace=False, max_expansions=max_expansions, time_limit_s=time_limit_s
    )

    gc.collect()  # so garbage from the timed run is not counted
    _record_expanded_order = False
    tracemalloc.start()
    try:
        solver(
            problem,
            trace=False,
            max_expansions=timed.nodes_expanded + 1,
            time_limit_s=float("inf"),
        )
        _current, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        _record_expanded_order = True

    timed.peak_memory_kb = peak_bytes / 1024
    return timed
