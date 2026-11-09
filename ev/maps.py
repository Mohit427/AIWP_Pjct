"""Map data: a tiny toy map, the hand-made corridor, and a seeded generator."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from math import ceil, sqrt

from ev.model import MAX_BATTERY_CAP, Problem, RoadMap

# One battery unit is taken to cover 30 km, so a full cap of 10 is about 300 km.
KM_PER_BATTERY_UNIT = 30

MAP_SIZES = {"small": 20, "medium": 60, "large": 150}
DEFAULT_SEEDS = {"small": 101, "medium": 202, "large": 303}

CHARGER_FRACTION = 0.25
# Average spacing between generated nodes. Area grows with the node count, so
# road lengths stay similar on every size and the battery caps stay meaningful.
KM_PER_NODE_SPACING = 25
NEAREST_LINKS_PER_NODE = 3
SEED_OFFSET_STEP = 1009  # prime, so offset seeds do not collide with nearby seeds
MAX_GENERATION_ATTEMPTS = 1000


def battery_cost_for_km(km: float) -> int:
    """Return whole battery units for a distance, rounding up.

    Rounding up is the safe direction: the car never arrives with less charge
    than the model claims. The price is some precision (see the limitations
    section of the notes).
    """
    return max(1, ceil(km / KM_PER_BATTERY_UNIT))


def is_feasible(
    road_map: RoadMap, start: str, destination: str, start_battery: int, battery_cap: int
) -> bool:
    """Return True if the destination is reachable from (start, start_battery).

    A small reachability sweep written here on purpose, so that map generation
    does not depend on the search algorithms that the maps are used to test.
    """
    problem = Problem(road_map, start, destination, start_battery, battery_cap)
    seen = {problem.initial_state()}
    queue = deque(seen)
    while queue:
        state = queue.popleft()
        if problem.is_goal(state):
            return True
        for _action, next_state, _cost in problem.successors(state):
            if next_state not in seen:
                seen.add(next_state)
                queue.append(next_state)
    return False


# --------------------------------------------------------------------------
# Toy map: small enough to trace by hand.
# --------------------------------------------------------------------------

TOY_START = "S"
TOY_GOAL = "G"


def toy_map() -> RoadMap:
    """Return a 4-node map for hand-traced examples.

            B
           / \\        S-A 30 min, 1 unit     A-G 30 min, 2 units
          S   G       S-B 45 min, 2 units    B-G 20 min, 1 unit
           \\ /        A has a charger: 10 min per unit
            A

    With 2 units at S, the route through B strands the car at B with an empty
    battery, even though B is one road from the goal.
    """
    road_map = RoadMap("toy")
    road_map.add_node("S", 0, 0)
    road_map.add_node("A", 30, 0)
    road_map.add_node("B", 30, 30)
    road_map.add_node("G", 60, 0)
    road_map.add_road("S", "A", 30, 30, 1)
    road_map.add_road("A", "G", 30, 30, 2)
    road_map.add_road("S", "B", 45, 45, 2)
    road_map.add_road("B", "G", 45, 20, 1)
    road_map.add_charger("A", 10, 20)
    return road_map


def toy_problem(start_battery: int = 2, battery_cap: int = 3) -> Problem:
    """Return the standard toy trip from S to G."""
    return Problem(toy_map(), TOY_START, TOY_GOAL, start_battery, battery_cap)


# --------------------------------------------------------------------------
# Hand-made corridor.
#
# NOTE: all coordinates, distances, times, charger locations and prices below
# are approximate and illustrative. They were estimated for this exercise and
# have NOT been verified against any survey or live data.
# --------------------------------------------------------------------------

CORRIDOR_START = "Chennai"
CORRIDOR_GOAL = "Bengaluru"
CORRIDOR_START_BATTERY = 6

# name: (x_km, y_km), roughly east-west and north-south from a point SW of Bengaluru.
_CORRIDOR_NODES = {
    "Chennai": (355, 119),
    "Sriperumbudur": (320, 107),
    "Arakkonam": (290, 119),
    "Kanchipuram": (293, 92),
    "Chengalpattu": (323, 76),
    "Tindivanam": (288, 27),
    "Ranipet": (253, 103),
    "Arani": (247, 74),
    "Tiruvannamalai": (225, 25),
    "Vellore": (231, 102),
    "Chittoor": (228, 135),
    "Palamaner": (190, 133),
    "Ambur": (187, 87),
    "Vaniyambadi": (176, 75),
    "Tirupattur": (170, 55),
    "Kuppam": (145, 83),
    "Krishnagiri": (131, 58),
    "Dharmapuri": (126, 14),
    "Kolar": (123, 126),
    "Hosur": (90, 82),
    "Hoskote": (87, 118),
    "Bengaluru": (64, 107),
}

# (a, b, road_km, minutes). Highways have a low minutes-per-km ratio and back
# roads a high one, so the fewest-roads route and the fastest route can differ.
_CORRIDOR_ROADS = [
    ("Chennai", "Sriperumbudur", 40, 50),
    ("Chennai", "Arakkonam", 70, 95),
    ("Chennai", "Chengalpattu", 60, 65),
    ("Sriperumbudur", "Kanchipuram", 35, 40),
    ("Sriperumbudur", "Ranipet", 80, 65),
    ("Sriperumbudur", "Arakkonam", 40, 55),
    ("Arakkonam", "Ranipet", 50, 70),
    ("Arakkonam", "Chittoor", 95, 125),
    ("Kanchipuram", "Ranipet", 50, 65),
    ("Kanchipuram", "Arani", 65, 90),
    ("Chengalpattu", "Kanchipuram", 38, 50),
    ("Chengalpattu", "Tindivanam", 65, 55),
    ("Tindivanam", "Tiruvannamalai", 85, 115),
    ("Tindivanam", "Arani", 75, 105),
    ("Ranipet", "Vellore", 25, 25),
    ("Arani", "Vellore", 38, 50),
    ("Arani", "Tiruvannamalai", 60, 85),
    ("Vellore", "Chittoor", 40, 55),
    ("Vellore", "Ambur", 52, 42),
    ("Chittoor", "Palamaner", 45, 60),
    ("Palamaner", "Kuppam", 75, 110),
    ("Palamaner", "Kolar", 80, 105),
    ("Ambur", "Vaniyambadi", 18, 15),
    ("Vaniyambadi", "Krishnagiri", 55, 45),
    ("Vaniyambadi", "Tirupattur", 25, 35),
    ("Vaniyambadi", "Kuppam", 45, 70),
    ("Tirupattur", "Krishnagiri", 42, 55),
    ("Tirupattur", "Tiruvannamalai", 85, 115),
    ("Tirupattur", "Dharmapuri", 65, 90),
    ("Tiruvannamalai", "Dharmapuri", 115, 160),
    ("Dharmapuri", "Krishnagiri", 48, 40),
    ("Dharmapuri", "Hosur", 90, 75),
    ("Krishnagiri", "Hosur", 50, 40),
    ("Kuppam", "Krishnagiri", 35, 50),
    ("Kuppam", "Kolar", 55, 80),
    ("Kolar", "Hoskote", 40, 40),
    ("Hoskote", "Bengaluru", 30, 50),
    ("Hosur", "Bengaluru", 40, 65),
    ("Hoskote", "Hosur", 42, 55),
]

# node: (minutes per unit, price per unit in rupees). Illustrative only.
_CORRIDOR_CHARGERS = {
    "Sriperumbudur": (15, 18),
    "Kanchipuram": (20, 16),
    "Vellore": (10, 22),
    "Tiruvannamalai": (20, 15),
    "Krishnagiri": (10, 24),
    "Kolar": (15, 18),
    "Hosur": (10, 22),
}


def corridor_map() -> RoadMap:
    """Build the Chennai to Bengaluru style map from the tables above."""
    road_map = RoadMap("corridor")
    for name, (x_km, y_km) in _CORRIDOR_NODES.items():
        road_map.add_node(name, x_km, y_km)
    for a, b, km, minutes in _CORRIDOR_ROADS:
        road_map.add_road(a, b, km, minutes, battery_cost_for_km(km))
    for node, (minutes_per_unit, price) in _CORRIDOR_CHARGERS.items():
        road_map.add_charger(node, minutes_per_unit, price)
    return road_map


def corridor_problem(
    start_battery: int = CORRIDOR_START_BATTERY,
    battery_cap: int = MAX_BATTERY_CAP,
    shuffle_seed: int | None = None,
) -> Problem:
    """Return the standard corridor trip."""
    return Problem(
        corridor_map(), CORRIDOR_START, CORRIDOR_GOAL, start_battery, battery_cap, shuffle_seed
    )


# --------------------------------------------------------------------------
# Seeded random generator.
# --------------------------------------------------------------------------


@dataclass
class GeneratedMap:
    """A generated map, its verified trip, and the seed that produced it."""

    road_map: RoadMap
    start: str
    destination: str
    start_battery: int
    battery_cap: int
    requested_seed: int
    seed_used: int  # differs from requested_seed if an earlier attempt was infeasible

    def problem(self, shuffle_seed: int | None = None) -> Problem:
        """Return this map's trip as a Problem."""
        return Problem(
            self.road_map,
            self.start,
            self.destination,
            self.start_battery,
            self.battery_cap,
            shuffle_seed,
        )


def _minimum_spanning_pairs(road_map: RoadMap, names: list[str]) -> set[tuple[str, str]]:
    """Return the pairs of a minimum spanning tree (Prim's algorithm).

    A spanning tree connects every node, which guarantees a connected map.
    A minimum one uses the shortest possible links, so the guarantee does not
    depend on a very long road that a small battery could never cover.
    """
    first, rest = names[0], names[1:]
    # For each node outside the tree: (distance to the tree, nearest tree node).
    closest = {name: (road_map.straight_line_km(name, first), first) for name in rest}
    pairs = set()
    while closest:
        # The name in the key breaks distance ties repeatably.
        name = min(closest, key=lambda candidate: (closest[candidate][0], candidate))
        _distance, parent = closest.pop(name)
        pairs.add((min(name, parent), max(name, parent)))
        for other, (best, _) in closest.items():
            distance = road_map.straight_line_km(other, name)
            if distance < best:
                closest[other] = (distance, name)
    return pairs


def _nearest_neighbour_pairs(road_map: RoadMap, names: list[str]) -> set[tuple[str, str]]:
    """Return links from each node to its nearest few nodes.

    A tree has exactly one route between any two places, which would make
    every algorithm trivially equal. These extra links create alternatives.
    """
    pairs = set()
    for name in names:
        others = sorted(
            (other for other in names if other != name),
            key=lambda other: (road_map.straight_line_km(name, other), other),
        )
        for other in others[:NEAREST_LINKS_PER_NODE]:
            pairs.add((min(name, other), max(name, other)))
    return pairs


def _build_random_map(num_nodes: int, seed: int) -> tuple[RoadMap, str, str]:
    """Build one candidate map and choose a start and goal. Feasibility not checked."""
    rng = random.Random(seed)
    side_km = sqrt(num_nodes) * KM_PER_NODE_SPACING
    road_map = RoadMap(f"random-{num_nodes}-seed{seed}")

    # Zero-padded names, so sorting by name matches numeric order.
    names = [f"N{i:03d}" for i in range(num_nodes)]
    for name in names:
        road_map.add_node(name, round(rng.uniform(0, side_km), 1), round(rng.uniform(0, side_km), 1))

    pairs = _minimum_spanning_pairs(road_map, names) | _nearest_neighbour_pairs(road_map, names)
    # Sorted because set order changes between Python runs, and each road below
    # draws random numbers: an unstable order would give a different map.
    for a, b in sorted(pairs):
        straight = road_map.straight_line_km(a, b)
        # Rounding UP keeps the road at least as long as the straight line.
        km = max(1, ceil(straight * rng.uniform(1.05, 1.35)))
        speed_km_per_hour = rng.uniform(45, 90)
        minutes = max(1, ceil(km / speed_km_per_hour * 60))
        road_map.add_road(a, b, km, minutes, battery_cost_for_km(km))

    for name in rng.sample(names, round(num_nodes * CHARGER_FRACTION)):
        road_map.add_charger(name, rng.choice([10, 15, 20]), rng.choice([15, 18, 22]))

    # The goal is drawn from the farther half of the map, so that trips are
    # long enough to be interesting rather than one or two hops.
    start = rng.choice(names)
    others = sorted(
        (name for name in names if name != start),
        key=lambda name: (-road_map.straight_line_km(start, name), name),
    )
    destination = rng.choice(others[: max(1, len(others) // 2)])
    return road_map, start, destination


def generate_map(
    num_nodes: int,
    seed: int,
    battery_cap: int = MAX_BATTERY_CAP,
    start_battery: int | None = None,
) -> GeneratedMap:
    """Return a connected random map whose trip is verified feasible.

    If a candidate is infeasible, the next one uses seed + attempt * offset.
    The whole retry sequence is determined by the seed, so results still repeat.

    start_battery defaults to half the cap, so that trips usually need charging.
    """
    if start_battery is None:
        start_battery = battery_cap // 2
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        seed_used = seed + attempt * SEED_OFFSET_STEP
        road_map, start, destination = _build_random_map(num_nodes, seed_used)
        if is_feasible(road_map, start, destination, start_battery, battery_cap):
            return GeneratedMap(
                road_map, start, destination, start_battery, battery_cap, seed, seed_used
            )
    raise RuntimeError(f"no feasible map after {MAX_GENERATION_ATTEMPTS} attempts")


def generate_sized_map(size: str, battery_cap: int = MAX_BATTERY_CAP) -> GeneratedMap:
    """Return the standard map for 'small', 'medium' or 'large' using its fixed seed."""
    return generate_map(MAP_SIZES[size], DEFAULT_SEEDS[size], battery_cap)
