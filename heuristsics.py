"""
heuristics.py
Author: ATK
Deliverable: Day 1 (Sun 8 Nov 2026) - Heuristics H0, H1, H2 for EV Trip Planner.
Designed to be imported directly by UCS, Greedy, A*, IDA*, and DFBB.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict

from ev.model import RoadMap, State


@dataclass(frozen=True)
class GraphBounds:
    """
    Precomputed global upper/lower bounds over the map to guarantee
    O(1) heuristic evaluation and strict admissibility/consistency.
    """
    v_max_km_per_min: float      # Max speed across all edges (km / min)
    km_per_battery_max: float    # Max distance achievable per 1 unit of battery
    min_charge_time_per_unit: int # Fastest charger time (min) per +1 battery unit


def precompute_graph_bounds(road_map: RoadMap) -> GraphBounds:
    """
    Scans the RoadMap once to find map-wide constants so H1 and H2
    do not have to scan edges during every node expansion.
    """
    v_max = 0.0
    km_per_bat = 0.0
    
    for neighbors in road_map.roads.values():
        for road in neighbors.values():
            if road.minutes > 0:
                v_max = max(v_max, road.km / road.minutes)
            if road.battery_cost > 0:
                km_per_bat = max(km_per_bat, road.km / road.battery_cost)
                
    min_charge = min(
        (charger.minutes_per_unit for charger in road_map.chargers.values()), 
        default=5
    )

    return GraphBounds(
        v_max_km_per_min=v_max,
        km_per_battery_max=km_per_bat,
        min_charge_time_per_unit=min_charge,
    )


# ---------------------------------------------------------------------------
# Core Heuristic Functions (H0, H1, H2)
# ---------------------------------------------------------------------------

def h0_zero(state: State, destination: str, road_map: RoadMap, bounds: GraphBounds) -> float:
    """
    H0: Zero heuristic (Uninformed baseline).
    With H0, A* reduces exactly to Uniform Cost Search (UCS).
    """
    return 0.0


def h1_straight_line_time(state: State, destination: str, road_map: RoadMap, bounds: GraphBounds) -> float:
    """
    H1: Straight-line distance to goal divided by maximum network speed (v_max).
    Admissible: Euclidean distance <= road distance, and v_max >= any edge speed.
    """
    node, _battery = state
    if node == destination:
        return 0.0
    
    d_sl = road_map.straight_line_km(node, destination)
    return d_sl / bounds.v_max_km_per_min


def h2_distance_plus_forced_charge(state: State, destination: str, road_map: RoadMap, bounds: GraphBounds) -> float:
    """
    H2: H1 + minimum possible charging time forced when straight-line distance
    exceeds current battery range.
    Dominates H1 while remaining admissible and consistent.
    """
    node, battery = state
    if node == destination:
        return 0.0

    d_sl = road_map.straight_line_km(node, destination)
    min_drive_time = d_sl / bounds.v_max_km_per_min

    # Minimum battery units needed to cover the straight-line distance
    min_units_needed = d_sl / bounds.km_per_battery_max
    energy_deficit = max(0.0, min_units_needed - float(battery))
    min_charge_time = energy_deficit * bounds.min_charge_time_per_unit

    return min_drive_time + min_charge_time


def make_heuristics(road_map: RoadMap, destination: str) -> Dict[str, Callable[[State], float]]:
    """
    Factory helper for informed search algorithms.
    Precomputes graph bounds once and returns single-argument functions `h(state)`.
    """
    bounds = precompute_graph_bounds(road_map)
    return {
        "H0": lambda state: h0_zero(state, destination, road_map, bounds),
        "H1": lambda state: h1_straight_line_time(state, destination, road_map, bounds),
        "H2": lambda state: h2_distance_plus_forced_charge(state, destination, road_map, bounds),
    }