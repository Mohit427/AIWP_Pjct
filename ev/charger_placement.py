"""Charger placement problem formulation and evaluation.

This module defines:
- The charger placement problem as a local search optimization
- Trip evaluation using Dijkstra's algorithm (via UCS)
- Brute-force optimum for small instances (ground truth)
- Sample trip generation
"""

from __future__ import annotations

import heapq
import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Callable

from ev.maps import RoadMap
from ev.metrics import Result, SearchRun, DEFAULT_MAX_EXPANSIONS, DEFAULT_TIME_LIMIT_S
from ev.model import Action, Problem, State, MAX_BATTERY_CAP, DRIVE, CHARGE


@dataclass(frozen=True)
class Trip:
    """A single trip to evaluate: start, goal, and starting battery."""
    start: str
    goal: str
    start_battery: int


@dataclass
class ChargerPlacementProblem:
    """The charger placement optimization problem.

    State: a frozenset of N charger locations chosen from M candidates.
    Objective: minimize average trip time over a fixed set of sample trips.
    Neighborhood: swap one chosen site with one unchosen site.
    """

    road_map: RoadMap
    candidates: list[str]          # All possible charger locations
    num_chargers: int              # Number of chargers to place (N)
    sample_trips: list[Trip]       # Fixed trips for evaluation
    start_battery: int = MAX_BATTERY_CAP // 2
    battery_cap: int = MAX_BATTERY_CAP
    unreachable_penalty: int = 1_000_000  # Large penalty for unreachable trips

    def __post_init__(self) -> None:
        if self.num_chargers > len(self.candidates):
            raise ValueError(f"Cannot place {self.num_chargers} chargers from {len(self.candidates)} candidates")
        if self.num_chargers < 1:
            raise ValueError("Must place at least 1 charger")
        # Validate all candidates exist in road map
        for c in self.candidates:
            if c not in self.road_map.coords:
                raise ValueError(f"Candidate {c} not in road map")
        # Validate all trip nodes exist
        for trip in self.sample_trips:
            if trip.start not in self.road_map.coords:
                raise ValueError(f"Trip start {trip.start} not in road map")
            if trip.goal not in self.road_map.coords:
                raise ValueError(f"Trip goal {trip.goal} not in road map")

    def initial_state(self, seed: int | None = None) -> frozenset[str]:
        """Return a random initial state (set of N charger locations)."""
        rng = random.Random(seed)
        return frozenset(rng.sample(self.candidates, self.num_chargers))

    def is_valid_state(self, state: frozenset[str]) -> bool:
        """Check if state has exactly N chargers from candidates."""
        return len(state) == self.num_chargers and state.issubset(set(self.candidates))

    def neighbors(self, state: frozenset[str]) -> list[frozenset[str]]:
        """Generate all neighboring states by swapping one charger.

        Each neighbor replaces one chosen site with one unchosen site.
        Number of neighbors = N * (M - N).
        """
        chosen = set(state)
        unchosen = set(self.candidates) - chosen
        neighbors = []
        for remove in chosen:
            for add in unchosen:
                new_state = frozenset((chosen - {remove}) | {add})
                neighbors.append(new_state)
        return neighbors

    def random_neighbor(self, state: frozenset[str], rng: random.Random) -> frozenset[str]:
        """Generate one random neighbor by swapping one charger."""
        chosen = list(state)
        unchosen = [c for c in self.candidates if c not in state]
        remove = rng.choice(chosen)
        add = rng.choice(unchosen)
        new_chosen = set(state)
        new_chosen.remove(remove)
        new_chosen.add(add)
        return frozenset(new_chosen)


def evaluate_charger_set(
    problem: ChargerPlacementProblem,
    charger_set: frozenset[str]
) -> float:
    """Evaluate a charger set: return average trip time over all sample trips.

    Uses Dijkstra (UCS) to find shortest trip time for each trip with the given chargers.
    """
    total_time = 0
    for trip in problem.sample_trips:
        trip_time = _dijkstra_trip_time(
            problem.road_map,
            charger_set,
            trip.start,
            trip.goal,
            trip.start_battery,
            problem.battery_cap,
        )
        if trip_time is None:
            total_time += problem.unreachable_penalty
        else:
            total_time += trip_time
    return total_time / len(problem.sample_trips)


def _dijkstra_trip_time(
    road_map: RoadMap,
    charger_set: frozenset[str],
    start: str,
    goal: str,
    start_battery: int,
    battery_cap: int,
) -> int | None:
    """Run Dijkstra/UCS to find minimum trip time with given chargers.

    State space: (node, battery_level)
    Actions: drive to neighbor (cost = time, battery -= cost), or charge 1 unit (cost = charge_time, battery += 1)
    Goal: reach goal node with any battery >= 0.

    The charger_set contains the NEW chargers to add. Existing chargers on the map
    are also available (they are not being removed).
    """
    # Merge existing chargers with new ones
    original_chargers = road_map.chargers
    # Combine existing chargers with the new charger set
    # New chargers use default charge rate/price if not specified
    combined_chargers = dict(original_chargers)
    # For new chargers not already in the map, add them with default values
    # (This assumes the problem's candidates are nodes that don't have chargers yet)
    for node in charger_set:
        if node not in combined_chargers:
            # Default charger: 15 min/unit, 20 rupees/unit
            from ev.model import Charger
            combined_chargers[node] = Charger(minutes_per_unit=15, price_per_unit=20)

    road_map.chargers = combined_chargers

    try:
        prob = Problem(road_map, start, goal, start_battery, battery_cap)
        return _ucs_trip_time(prob)
    finally:
        road_map.chargers = original_chargers


def _ucs_trip_time(problem: Problem) -> int | None:
    """Uniform Cost Search to find shortest trip time. Returns minutes or None if unreachable."""
    start_state = problem.initial_state()
    if problem.is_goal(start_state):
        return 0

    # Priority queue: (total_cost, state, path)
    # We don't need the full path, just the cost
    frontier = [(0, start_state)]
    visited: dict[State, int] = {start_state: 0}  # state -> best cost so far

    while frontier:
        cost, state = heapq.heappop(frontier)

        # If we already found a better path to this state, skip
        if visited.get(state, float('inf')) < cost:
            continue

        if problem.is_goal(state):
            return cost

        for action, next_state, step_cost in problem.successors(state):
            new_cost = cost + step_cost
            if new_cost < visited.get(next_state, float('inf')):
                visited[next_state] = new_cost
                heapq.heappush(frontier, (new_cost, next_state))

    return None  # Unreachable


def generate_sample_trips(
    road_map: RoadMap,
    num_trips: int,
    seed: int = 42,
    start_battery: int | None = None,
    battery_cap: int = MAX_BATTERY_CAP,
) -> list[Trip]:
    """Generate a fixed set of sample trips for evaluation.

    Trips are generated by picking random start/goal pairs that are feasible.
    """
    if start_battery is None:
        start_battery = battery_cap // 2

    rng = random.Random(seed)
    nodes = list(road_map.coords.keys())
    trips = []

    # Use the feasibility check from maps
    from ev.maps import is_feasible

    attempts = 0
    max_attempts = num_trips * 10
    while len(trips) < num_trips and attempts < max_attempts:
        start = rng.choice(nodes)
        goal = rng.choice([n for n in nodes if n != start])
        if is_feasible(road_map, start, goal, start_battery, battery_cap):
            trips.append(Trip(start, goal, start_battery))
        attempts += 1

    if len(trips) < num_trips:
        # Fallback: just use first feasible pairs
        for start in nodes:
            for goal in nodes:
                if start != goal and len(trips) < num_trips:
                    if is_feasible(road_map, start, goal, start_battery, battery_cap):
                        trips.append(Trip(start, goal, start_battery))

    return trips


@dataclass
class BruteForceResult:
    """Result of brute-force search over all combinations."""
    optimal_state: frozenset[str]
    optimal_value: float
    all_evaluated: dict[frozenset[str], float] = field(default_factory=dict)
    runtime_s: float = 0.0


def brute_force_optimum(problem: ChargerPlacementProblem) -> BruteForceResult:
    """Exhaustively evaluate all combinations to find true global optimum.

    Only feasible for small M, N (e.g., M <= 15, N <= 5).
    """
    start_time = time.perf_counter()
    best_value = float('inf')
    best_state = None
    all_evaluated = {}

    for combo in itertools.combinations(problem.candidates, problem.num_chargers):
        state = frozenset(combo)
        value = evaluate_charger_set(problem, state)
        all_evaluated[state] = value
        if value < best_value:
            best_value = value
            best_state = state

    runtime = time.perf_counter() - start_time
    return BruteForceResult(
        optimal_state=best_state,
        optimal_value=best_value,
        all_evaluated=all_evaluated,
        runtime_s=runtime,
    )


# -------------------------------------------------------------------------
# Local Search Result Classes
# -------------------------------------------------------------------------

@dataclass
class LocalSearchResult:
    """Base result for local search algorithms."""
    algorithm: str
    best_state: frozenset[str]
    best_value: float
    iterations: int
    runtime_s: float
    trace_history: list[dict] = field(default_factory=list)  # For visualization


@dataclass
class HillClimbingResult(LocalSearchResult):
    """Result for hill climbing variants."""
    final_state: frozenset[str] = field(init=False)
    stopped_reason: str = ""  # "local_optimum", "max_iterations", "no_improvement"

    def __post_init__(self):
        self.final_state = self.best_state


@dataclass
class RandomRestartResult(LocalSearchResult):
    """Result for random-restart hill climbing."""
    all_runs: list[HillClimbingResult] = field(default_factory=list)
    num_restarts: int = 0


@dataclass
class SimulatedAnnealingResult(LocalSearchResult):
    """Result for simulated annealing."""
    final_temp: float = 0.0
    acceptance_rate: float = 0.0
    cooling_schedule: str = ""