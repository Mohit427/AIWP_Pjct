"""Heuristics H0, H1, and H2 for informed search (UCS, Greedy, A*, IDA*, DFBB).

Author: ATK

Mathematical Formulation & Proofs
---------------------------------
Let s = (u, b) be the current state at node u with integer battery b, and let
g = destination. Let d_SL(u, g) be the Euclidean straight-line distance (km).

Precomputed map-wide constants (GraphBounds):
  - v_max = max_{(x,y) in E} (road.km / road.minutes)
  - eta_max = max_{(x,y) in E} (road.km / road.battery_cost)
  - tau_min = min_{c in Chargers} (c.minutes_per_unit)

1. H0(s) = 0
   - Admissible & Consistent: 0 <= h*(s) and 0 <= cost(s, a, s') + 0 since all
     action costs are >= 1 minute. With H0, A* is identical to UCS.

2. H1(s) = d_SL(u, g) / v_max
   - Admissibility: RoadMap.add_road enforces road.km >= d_SL(x, y) on every
     edge. By the Euclidean triangle inequality, any path from u to g has total
     road length L >= d_SL(u, g). Since no road is faster than v_max, driving
     time alone is at least L / v_max >= d_SL(u, g) / v_max.
   - Consistency:
     * Drive (u -> v): H1(u, b) - H1(v, b') = (d_SL(u, g) - d_SL(v, g)) / v_max
       <= d_SL(u, v) / v_max <= road.km / v_max <= road.minutes.
     * Charge at u: location is unchanged, so H1(u, b) - H1(u, b+1) = 0 <= tau(u).

3. H2(s) = H1(s) + tau_min * max(0, d_SL(u, g) / eta_max - b)
   - Admissibility: Covering distance L >= d_SL(u, g) consumes at least
     L / eta_max >= d_SL(u, g) / eta_max battery units. With current battery b,
     at least max(0, d_SL(u, g) / eta_max - b) units must be charged along the
     trip, each taking at least tau_min minutes. Since driving and charging
     times are disjoint action costs, H1(s) + min_charge_time <= h*(s).
   - Consistency:
     * Drive (u -> v) costing db = b - b' units: Since db >= road.km / eta_max
       >= (d_SL(u, g) - d_SL(v, g)) / eta_max, the energy deficit at v is at
       least the energy deficit at u. Thus the forced-charge term never
       decreases on a drive step, and H2(s) - H2(s') <= H1(s) - H1(s') <= road.minutes.
     * Charge at u (b' = b + 1): H1 is unchanged, and the energy deficit drops
       by at most 1 unit, reducing H2 by at most tau_min <= charger.minutes_per_unit.
   - Dominance: H2(s) >= H1(s) >= H0(s) for all states s.
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