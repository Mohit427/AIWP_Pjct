"""Heuristics H0, H1, and H2 for informed search (UCS, Greedy, A*, IDA*, DFBB).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ev.model import Problem, RoadMap, State

HeuristicFn = Callable[[State], float]


@dataclass(frozen=True)
class GraphBounds:
    """Precomputed map-wide bounds used by H1 and H2 in O(1) time."""

    v_max_km_per_min: float
    km_per_battery_max: float
    min_charge_time_per_unit: int


def precompute_graph_bounds(road_map: RoadMap) -> GraphBounds:
    """Scan all roads and chargers once to extract global upper/lower bounds."""
    v_max = 0.0
    km_per_bat = 0.0

    for neighbors in road_map.roads.values():
        for road in neighbors.values():
            if road.minutes > 0:
                v_max = max(v_max, road.km / road.minutes)
            if road.battery_cost > 0:
                km_per_bat = max(km_per_bat, road.km / road.battery_cost)

    # Safe fallbacks for degenerate 1-node maps with no roads or no chargers.
    if v_max <= 0.0:
        v_max = 1.0
    if km_per_bat <= 0.0:
        km_per_bat = 30.0

    min_charge = min(
        (charger.minutes_per_unit for charger in road_map.chargers.values()),
        default=0,
    )

    return GraphBounds(
        v_max_km_per_min=v_max,
        km_per_battery_max=km_per_bat,
        min_charge_time_per_unit=min_charge,
    )


def h0_zero(
    state: State,
    destination: str,
    road_map: RoadMap,
    bounds: GraphBounds | None = None,
) -> float:
    """H0: Zero baseline heuristic (reduces A* to Uniform Cost Search)."""
    return 0.0


def h1_straight_line_time(
    state: State,
    destination: str,
    road_map: RoadMap,
    bounds: GraphBounds | None = None,
) -> float:
    """H1: Straight-line distance to destination divided by max road speed."""
    node, _battery = state
    if node == destination:
        return 0.0
    if bounds is None:
        bounds = precompute_graph_bounds(road_map)
    d_sl = road_map.straight_line_km(node, destination)
    return d_sl / bounds.v_max_km_per_min


def h2_distance_plus_forced_charge(
    state: State,
    destination: str,
    road_map: RoadMap,
    bounds: GraphBounds | None = None,
) -> float:
    """H2: H1 plus minimum charging time forced when destination exceeds range."""
    node, battery = state
    if node == destination:
        return 0.0
    if bounds is None:
        bounds = precompute_graph_bounds(road_map)

    d_sl = road_map.straight_line_km(node, destination)
    min_drive_time = d_sl / bounds.v_max_km_per_min

    min_units_needed = d_sl / bounds.km_per_battery_max
    energy_deficit = max(0.0, min_units_needed - float(battery))
    min_charge_time = energy_deficit * bounds.min_charge_time_per_unit

    return min_drive_time + min_charge_time


def make_heuristics(
    road_map: RoadMap, destination: str
) -> dict[str, HeuristicFn]:
    """Return {'H0': fn, 'H1': fn, 'H2': fn} bound to road_map and destination."""
    bounds = precompute_graph_bounds(road_map)
    return {
        "H0": lambda state: h0_zero(state, destination, road_map, bounds),
        "H1": lambda state: h1_straight_line_time(state, destination, road_map, bounds),
        "H2": lambda state: h2_distance_plus_forced_charge(
            state, destination, road_map, bounds
        ),
    }


def make_problem_heuristics(problem: Problem) -> dict[str, HeuristicFn]:
    """Convenience wrapper taking a Problem instance directly."""
    return make_heuristics(problem.road_map, problem.destination)


HEURISTIC_NAMES = ("H0", "H1", "H2")