"""Local search algorithms for charger placement optimization.

Implements:
- Steepest-Ascent Hill Climbing
- First-Choice Hill Climbing
- Random-Restart Hill Climbing
- Simulated Annealing (exponential and logarithmic cooling)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable

from ev.charger_placement import (
    ChargerPlacementProblem,
    HillClimbingResult,
    LocalSearchResult,
    RandomRestartResult,
    SimulatedAnnealingResult,
    evaluate_charger_set,
)


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def _trace_step(
    trace: bool,
    trace_history: list[dict],
    iteration: int,
    current_value: float,
    current_state: frozenset[str],
    neighbor_value: float | None = None,
    neighbor_state: frozenset[str] | None = None,
    accepted: bool | None = None,
    extra: dict | None = None,
) -> None:
    """Record a trace step if tracing is enabled."""
    if not trace:
        return
    step = {
        "iteration": iteration,
        "current_value": current_value,
        "current_state": set(current_state),
    }
    if neighbor_value is not None:
        step["neighbor_value"] = neighbor_value
        step["neighbor_state"] = set(neighbor_state) if neighbor_state else None
    if accepted is not None:
        step["accepted"] = accepted
    if extra:
        step.update(extra)
    trace_history.append(step)


# -------------------------------------------------------------------------
# Steepest-Ascent Hill Climbing
# -------------------------------------------------------------------------

def steepest_ascent_hill_climbing(
    problem: ChargerPlacementProblem,
    initial_state: frozenset[str] | None = None,
    max_iterations: int = 1000,
    trace: bool = False,
    seed: int | None = None,
) -> HillClimbingResult:
    """Steepest-ascent hill climbing: evaluate ALL neighbors, pick the best.

    At each step, generates all N*(M-N) neighbors and moves to the best one
    if it improves the objective. Stops when no neighbor is better.
    """
    rng = random.Random(seed)
    start_time = time.perf_counter()

    # Initialize
    if initial_state is None:
        current_state = problem.initial_state(seed)
    else:
        current_state = initial_state

    current_value = evaluate_charger_set(problem, current_state)
    best_state = current_state
    best_value = current_value
    trace_history: list[dict] = []

    _trace_step(trace, trace_history, 0, current_value, current_state, extra={"phase": "initial"})

    for iteration in range(1, max_iterations + 1):
        # Generate all neighbors
        neighbors = problem.neighbors(current_state)
        if not neighbors:
            stopped_reason = "no_neighbors"
            break

        # Evaluate all neighbors
        neighbor_values = [(n, evaluate_charger_set(problem, n)) for n in neighbors]

        # Find best neighbor
        best_neighbor, best_neighbor_value = min(neighbor_values, key=lambda x: x[1])

        _trace_step(
            trace, trace_history, iteration,
            current_value, current_state,
            best_neighbor_value, best_neighbor,
            accepted=False,
            extra={"phase": "evaluate", "num_neighbors": len(neighbors)}
        )

        # Check for improvement
        if best_neighbor_value < current_value:
            # Move to better neighbor
            current_state = best_neighbor
            current_value = best_neighbor_value
            if current_value < best_value:
                best_value = current_value
                best_state = current_state

            _trace_step(
                trace, trace_history, iteration,
                current_value, current_state,
                extra={"phase": "move", "improvement": True}
            )
        else:
            # No improving neighbor found - local optimum
            stopped_reason = "local_optimum"
            _trace_step(
                trace, trace_history, iteration,
                current_value, current_state,
                extra={"phase": "stop", "reason": "local_optimum"}
            )
            break
    else:
        # Loop completed without break
        stopped_reason = "max_iterations"

    runtime = time.perf_counter() - start_time

    return HillClimbingResult(
        algorithm="Steepest-Ascent Hill Climbing",
        best_state=best_state,
        best_value=best_value,
        iterations=iteration,
        runtime_s=runtime,
        trace_history=trace_history,
        stopped_reason=stopped_reason,
    )


# -------------------------------------------------------------------------
# First-Choice Hill Climbing
# -------------------------------------------------------------------------

def first_choice_hill_climbing(
    problem: ChargerPlacementProblem,
    initial_state: frozenset[str] | None = None,
    max_iterations: int = 1000,
    max_neighbors_per_iter: int = 50,
    trace: bool = False,
    seed: int | None = None,
) -> HillClimbingResult:
    """First-choice hill climbing: randomly sample neighbors until finding improvement.

    At each iteration, randomly generates up to K neighbors and evaluates them
    one at a time. Moves to the first neighbor that improves the objective.
    If no improvement after K tries, stops.
    """
    rng = random.Random(seed)
    start_time = time.perf_counter()

    # Initialize
    if initial_state is None:
        current_state = problem.initial_state(seed)
    else:
        current_state = initial_state

    current_value = evaluate_charger_set(problem, current_state)
    best_state = current_state
    best_value = current_value
    trace_history: list[dict] = []

    _trace_step(trace, trace_history, 0, current_value, current_state, extra={"phase": "initial"})

    for iteration in range(1, max_iterations + 1):
        improved = False
        neighbors_checked = 0

        for _ in range(max_neighbors_per_iter):
            neighbor_state = problem.random_neighbor(current_state, rng)
            neighbor_value = evaluate_charger_set(problem, neighbor_state)
            neighbors_checked += 1

            _trace_step(
                trace, trace_history, iteration,
                current_value, current_state,
                neighbor_value, neighbor_state,
                accepted=False,
                extra={"phase": "evaluate", "neighbors_checked": neighbors_checked}
            )

            if neighbor_value < current_value:
                # Found improvement - move to it
                current_state = neighbor_state
                current_value = neighbor_value
                if current_value < best_value:
                    best_value = current_value
                    best_state = current_state
                improved = True

                _trace_step(
                    trace, trace_history, iteration,
                    current_value, current_state,
                    extra={"phase": "move", "improvement": True, "neighbors_checked": neighbors_checked}
                )
                break

        if not improved:
            # No improvement after checking max_neighbors_per_iter neighbors
            stopped_reason = "no_improvement"
            _trace_step(
                trace, trace_history, iteration,
                current_value, current_state,
                extra={"phase": "stop", "reason": "no_improvement", "neighbors_checked": neighbors_checked}
            )
            break
    else:
        stopped_reason = "max_iterations"

    runtime = time.perf_counter() - start_time

    return HillClimbingResult(
        algorithm="First-Choice Hill Climbing",
        best_state=best_state,
        best_value=best_value,
        iterations=iteration,
        runtime_s=runtime,
        trace_history=trace_history,
        stopped_reason=stopped_reason,
    )


# -------------------------------------------------------------------------
# Random-Restart Hill Climbing
# -------------------------------------------------------------------------

def random_restart_hill_climbing(
    problem: ChargerPlacementProblem,
    num_restarts: int = 20,
    use_first_choice: bool = False,
    max_iterations_per_run: int = 1000,
    max_neighbors_per_iter: int = 50,
    trace: bool = False,
    seed: int | None = None,
) -> RandomRestartResult:
    """Random-restart hill climbing: run hill climbing from multiple random starts.

    Returns the best result across all restarts. Each restart uses a different
    random seed for reproducibility.
    """
    rng = random.Random(seed)
    start_time = time.perf_counter()

    base_seed = seed if seed is not None else rng.randint(0, 2**31 - 1)
    all_runs: list[HillClimbingResult] = []
    best_state = None
    best_value = float('inf')
    trace_history: list[dict] = []

    if use_first_choice:
        def run_hill_climb(**kwargs):
            return first_choice_hill_climbing(
                problem,
                max_iterations=max_iterations_per_run,
                max_neighbors_per_iter=max_neighbors_per_iter,
                trace=False,
                **kwargs
            )
    else:
        def run_hill_climb(**kwargs):
            return steepest_ascent_hill_climbing(
                problem,
                max_iterations=max_iterations_per_run,
                trace=False,
                **kwargs
            )

    for restart in range(num_restarts):
        restart_seed = base_seed + restart * 1009  # Prime offset for independence

        if trace:
            trace_history.append({
                "restart": restart,
                "seed": restart_seed,
                "phase": "restart_start"
            })

        result = run_hill_climb(seed=restart_seed)

        all_runs.append(result)

        if result.best_value < best_value:
            best_value = result.best_value
            best_state = result.best_state

        if trace:
            trace_history.append({
                "restart": restart,
                "best_value": result.best_value,
                "best_state": set(result.best_state),
                "iterations": result.iterations,
                "stopped_reason": result.stopped_reason,
                "phase": "restart_end"
            })

    runtime = time.perf_counter() - start_time

    return RandomRestartResult(
        algorithm=f"Random-Restart {'First-Choice' if use_first_choice else 'Steepest-Ascent'} Hill Climbing",
        best_state=best_state,
        best_value=best_value,
        iterations=sum(r.iterations for r in all_runs),
        runtime_s=runtime,
        trace_history=trace_history,
        all_runs=all_runs,
        num_restarts=num_restarts,
    )


# -------------------------------------------------------------------------
# Simulated Annealing
# -------------------------------------------------------------------------

def _exponential_cooling(initial_temp: float, alpha: float, iteration: int) -> float:
    """Exponential (geometric) cooling: T = T0 * alpha^iteration."""
    return initial_temp * (alpha ** iteration)


def _logarithmic_cooling(initial_temp: float, iteration: int) -> float:
    """Logarithmic cooling: T = T0 / log(1 + iteration)."""
    import math
    if iteration <= 0:
        return initial_temp
    return initial_temp / math.log(1 + iteration)


def simulated_annealing(
    problem: ChargerPlacementProblem,
    initial_state: frozenset[str] | None = None,
    initial_temp: float = 100.0,
    min_temp: float = 0.01,
    cooling_schedule: str = "exponential",  # "exponential" or "logarithmic"
    alpha: float = 0.95,  # For exponential cooling
    max_iterations: int = 10000,
    trace: bool = False,
    seed: int | None = None,
) -> SimulatedAnnealingResult:
    """Simulated annealing for charger placement.

    Accepts worse moves with probability exp(-ΔE / T) to escape local optima.
    Two cooling schedules supported: exponential and logarithmic.
    """
    import math

    rng = random.Random(seed)
    start_time = time.perf_counter()

    # Initialize
    if initial_state is None:
        current_state = problem.initial_state(seed)
    else:
        current_state = initial_state

    current_value = evaluate_charger_set(problem, current_state)
    best_state = current_state
    best_value = current_value

    trace_history: list[dict] = []
    accepted_count = 0
    total_proposals = 0

    _trace_step(trace, trace_history, 0, current_value, current_state,
                extra={"phase": "initial", "temperature": initial_temp})

    # Select cooling function
    if cooling_schedule == "exponential":
        cooling_fn = lambda it: _exponential_cooling(initial_temp, alpha, it)
    elif cooling_schedule == "logarithmic":
        cooling_fn = lambda it: _logarithmic_cooling(initial_temp, it)
    else:
        raise ValueError(f"Unknown cooling schedule: {cooling_schedule}")

    for iteration in range(1, max_iterations + 1):
        temperature = cooling_fn(iteration)

        if temperature < min_temp:
            stopped_reason = "min_temp_reached"
            _trace_step(trace, trace_history, iteration, current_value, current_state,
                        extra={"phase": "stop", "reason": "min_temp_reached", "temperature": temperature})
            break

        # Generate random neighbor
        neighbor_state = problem.random_neighbor(current_state, rng)
        neighbor_value = evaluate_charger_set(problem, neighbor_state)
        total_proposals += 1

        delta_e = neighbor_value - current_value  # Positive = worse

        # Acceptance decision
        if delta_e < 0:
            # Improvement - always accept
            accepted = True
        else:
            # Worse - accept with probability exp(-ΔE / T)
            prob = math.exp(-delta_e / temperature)
            accepted = rng.random() < prob

        _trace_step(trace, trace_history, iteration, current_value, current_state,
                    neighbor_value, neighbor_state, accepted,
                    extra={
                        "phase": "propose",
                        "temperature": temperature,
                        "delta_e": delta_e,
                        "accept_prob": math.exp(-delta_e / temperature) if delta_e > 0 else 1.0,
                    })

        if accepted:
            current_state = neighbor_state
            current_value = neighbor_value
            accepted_count += 1

            if current_value < best_value:
                best_value = current_value
                best_state = current_state

            _trace_step(trace, trace_history, iteration, current_value, current_state,
                        extra={"phase": "accept", "temperature": temperature})
    else:
        stopped_reason = "max_iterations"

    runtime = time.perf_counter() - start_time
    acceptance_rate = accepted_count / total_proposals if total_proposals > 0 else 0.0
    final_temp = cooling_fn(iteration)

    return SimulatedAnnealingResult(
        algorithm=f"Simulated Annealing ({cooling_schedule})",
        best_state=best_state,
        best_value=best_value,
        iterations=iteration,
        runtime_s=runtime,
        trace_history=trace_history,
        final_temp=final_temp,
        acceptance_rate=acceptance_rate,
        cooling_schedule=cooling_schedule,
    )


# -------------------------------------------------------------------------
# Algorithm Registry (for easy import in main.py)
# -------------------------------------------------------------------------
# Note: Random Walk is implemented by Rohiit (separate module)
# Import it from there when available: from ev.rohiit_algorithms import random_walk

LOCAL_SEARCH_ALGORITHMS = {
    "steepest_ascent": steepest_ascent_hill_climbing,
    "first_choice": first_choice_hill_climbing,
    "random_restart_steepest": lambda **kw: random_restart_hill_climbing(**kw, use_first_choice=False),
    "random_restart_first_choice": lambda **kw: random_restart_hill_climbing(**kw, use_first_choice=True),
    "simulated_annealing_exp": lambda **kw: simulated_annealing(**kw, cooling_schedule="exponential"),
    "simulated_annealing_log": lambda **kw: simulated_annealing(**kw, cooling_schedule="logarithmic"),
}