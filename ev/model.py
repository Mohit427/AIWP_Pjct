"""The shared search model. Every algorithm in the project codes against this.

Formulation
-----------
State    (node, battery): where the car is and how many battery units it has.
         Elapsed time is deliberately NOT in the state. It is the path cost.
         If time were part of the state, driving around a loop would keep
         producing "new" states forever and the state space would be infinite.
         Without it there are at most (nodes x (cap + 1)) states.
Initial  (start, start_battery).
Goal     node == destination, with any battery level >= 0.
Actions  drive to a neighbour (needs battery >= the road's battery cost), or
         charge 1 unit (only at a charger node, and only below the cap).
Cost     minutes: the road's minutes, or the charger's minutes per unit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import hypot
from typing import NamedTuple

MAX_BATTERY_CAP = 10

DRIVE = "drive"
CHARGE = "charge"

# A state is a plain tuple so it is hashable and can go straight into a set.
State = tuple[str, int]


class Action(NamedTuple):
    """One step of a plan, e.g. Action("drive", "Vellore")."""

    kind: str  # DRIVE or CHARGE
    target: str  # the node driven to, or the node charged at

    def __str__(self) -> str:
        """Short readable form used in trace output and printed plans."""
        return f"{self.kind} {self.target}"


@dataclass(frozen=True)
class Road:
    """The attributes of one road. Frozen because maps must not change mid-search."""

    km: float
    minutes: int
    battery_cost: int


@dataclass(frozen=True)
class Charger:
    """The attributes of one charging station."""

    minutes_per_unit: int
    price_per_unit: int


class RoadMap:
    """An undirected road network with coordinates and chargers.

    The add_* methods reject bad data immediately, because a broken map would
    otherwise show up much later as a confusing search result.
    """

    def __init__(self, name: str) -> None:
        """Create an empty map. The name is only used for display."""
        self.name = name
        self.coords: dict[str, tuple[float, float]] = {}
        self.roads: dict[str, dict[str, Road]] = {}
        self.chargers: dict[str, Charger] = {}

    def add_node(self, name: str, x_km: float, y_km: float) -> None:
        """Add a place at (x_km, y_km)."""
        if name in self.coords:
            raise ValueError(f"duplicate node: {name}")
        self.coords[name] = (x_km, y_km)
        self.roads[name] = {}

    def straight_line_km(self, a: str, b: str) -> float:
        """Return the straight-line distance between two nodes.

        Teammates' heuristics use this as a lower bound on the road distance.
        """
        ax, ay = self.coords[a]
        bx, by = self.coords[b]
        return hypot(ax - bx, ay - by)

    def add_road(self, a: str, b: str, km: float, minutes: int, battery_cost: int) -> None:
        """Add a two-way road after checking the facts search depends on.

        - minutes >= 1: every step has positive cost, which uniform-cost
          reasoning relies on.
        - battery_cost >= 1 and an int: the battery is a whole number of units.
        - km >= straight-line distance: otherwise a straight-line heuristic
          could overestimate and A* would lose its optimality guarantee.
        """
        for node in (a, b):
            if node not in self.coords:
                raise ValueError(f"unknown node: {node}")
        if a == b:
            raise ValueError(f"road from {a} to itself")
        if b in self.roads[a]:
            raise ValueError(f"duplicate road: {a} - {b}")
        if not isinstance(minutes, int) or minutes < 1:
            raise ValueError(f"{a} - {b}: minutes must be an int >= 1")
        if not isinstance(battery_cost, int) or battery_cost < 1:
            raise ValueError(f"{a} - {b}: battery cost must be an int >= 1")
        if km < self.straight_line_km(a, b):
            raise ValueError(f"{a} - {b}: road is shorter than the straight line")
        road = Road(km, minutes, battery_cost)
        self.roads[a][b] = road
        self.roads[b][a] = road

    def add_charger(self, node: str, minutes_per_unit: int, price_per_unit: int) -> None:
        """Mark a node as a charging station."""
        if node not in self.coords:
            raise ValueError(f"unknown node: {node}")
        if not isinstance(minutes_per_unit, int) or minutes_per_unit < 1:
            raise ValueError(f"{node}: minutes per unit must be an int >= 1")
        self.chargers[node] = Charger(minutes_per_unit, price_per_unit)

    def num_nodes(self) -> int:
        """Return the number of places on the map."""
        return len(self.coords)

    def num_roads(self) -> int:
        """Return the number of two-way roads (each is stored twice, once per end)."""
        return sum(len(links) for links in self.roads.values()) // 2


class Problem:
    """One trip on one map: the object every search algorithm receives."""

    def __init__(
        self,
        road_map: RoadMap,
        start: str,
        destination: str,
        start_battery: int,
        battery_cap: int = MAX_BATTERY_CAP,
        shuffle_seed: int | None = None,
    ) -> None:
        """Validate the trip and pre-sort each node's options.

        Sorting once here makes successors() fast, and it is what makes the
        default successor order deterministic.

        shuffle_seed: None keeps the sorted order. An int gives a repeatable
        random order, used to test how much an algorithm depends on order.
        """
        for node in (start, destination):
            if node not in road_map.coords:
                raise ValueError(f"unknown node: {node}")
        if not 1 <= battery_cap <= MAX_BATTERY_CAP:
            raise ValueError(f"battery cap must be in 1..{MAX_BATTERY_CAP}")
        if not 0 <= start_battery <= battery_cap:
            raise ValueError("start battery must be in 0..battery cap")

        self.road_map = road_map
        self.start = start
        self.destination = destination
        self.start_battery = start_battery
        self.battery_cap = battery_cap
        self.shuffle_seed = shuffle_seed

        self._drives = {
            node: [
                (Action(DRIVE, nbr), nbr, links[nbr].minutes, links[nbr].battery_cost)
                for nbr in sorted(links)
            ]
            for node, links in road_map.roads.items()
        }
        self._charges = {
            node: (Action(CHARGE, node), charger.minutes_per_unit)
            for node, charger in road_map.chargers.items()
        }

    def initial_state(self) -> State:
        """Return the state the trip begins in."""
        return (self.start, self.start_battery)

    def is_goal(self, state: State) -> bool:
        """Return True at the destination. Any battery level counts as arrived."""
        return state[0] == self.destination

    def successors(self, state: State) -> list[tuple[Action, State, int]]:
        """Return every legal (action, next_state, step_cost) from a state.

        Order: drives sorted by neighbour name, then charge, so all runs repeat
        exactly.

        The shuffle is seeded from (seed, state) rather than drawn from one
        running generator. A state therefore gets the same order every time it
        is expanded, however many calls came before. IDDFS needs this, because
        it re-expands the same states on every iteration.
        """
        node, battery = state
        options = [
            (action, (nbr, battery - cost), minutes)
            for action, nbr, minutes, cost in self._drives[node]
            if battery >= cost
        ]
        charge = self._charges.get(node)
        if charge is not None and battery < self.battery_cap:
            action, minutes = charge
            options.append((action, (node, battery + 1), minutes))
        if self.shuffle_seed is not None:
            random.Random(f"{self.shuffle_seed}:{node}:{battery}").shuffle(options)
        return options

    def replay(self, path: list[Action]) -> tuple[State, int]:
        """Execute a plan from the initial state and return (final_state, minutes).

        Raises ValueError on an illegal step. This re-checks a returned path
        using only the model, so a bug inside a search algorithm cannot make
        its own answer look valid.
        """
        state = self.initial_state()
        total_minutes = 0
        for step, action in enumerate(path):
            for candidate, next_state, cost in self.successors(state):
                if candidate == action:
                    state = next_state
                    total_minutes += cost
                    break
            else:
                raise ValueError(f"step {step}: {action} is illegal from {state}")
        return state, total_minutes
