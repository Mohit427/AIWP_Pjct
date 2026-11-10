"""Tests for the shared model and the map data."""

import unittest
from collections import deque

from ev.maps import (
    corridor_map,
    corridor_problem,
    generate_map,
    is_feasible,
    seeded_start_goal_pairs,
    toy_map,
    toy_problem,
)
from ev.model import Action, Problem, RoadMap


def fingerprint(generated):
    """Return a comparable snapshot of a generated map."""
    road_map = generated.road_map
    return (road_map.coords, road_map.roads, road_map.chargers, generated.start, generated.destination)


class ModelTests(unittest.TestCase):
    def test_drive_before_charge_sorted_by_name(self):
        options = toy_problem().successors(("A", 2))
        self.assertEqual(
            [action for action, _next, _cost in options],
            [Action("drive", "G"), Action("drive", "S"), Action("charge", "A")],
        )

    def test_drive_needs_enough_battery(self):
        kinds = [action for action, _n, _c in toy_problem().successors(("A", 1))]
        self.assertNotIn(Action("drive", "G"), kinds)

    def test_no_charge_at_cap_or_without_charger(self):
        problem = toy_problem()
        self.assertNotIn(
            Action("charge", "A"), [a for a, _n, _c in problem.successors(("A", 3))]
        )
        self.assertEqual(problem.successors(("B", 0)), [])

    def test_shuffle_is_repeatable_and_keeps_the_same_options(self):
        plain = corridor_problem()
        shuffled = corridor_problem(shuffle_seed=3)
        state = ("Vellore", 5)
        self.assertEqual(shuffled.successors(state), shuffled.successors(state))
        self.assertCountEqual(shuffled.successors(state), plain.successors(state))

    def test_replay_rejects_illegal_action(self):
        with self.assertRaises(ValueError):
            toy_problem().replay([Action("drive", "G")])

    def test_invalid_start_battery_rejected(self):
        with self.assertRaises(ValueError):
            Problem(toy_map(), "S", "G", 4, 3)

    def test_road_shorter_than_straight_line_rejected(self):
        road_map = RoadMap("bad")
        road_map.add_node("P", 0, 0)
        road_map.add_node("Q", 100, 0)
        with self.assertRaises(ValueError):
            road_map.add_road("P", "Q", 50, 10, 1)


class MapTests(unittest.TestCase):
    def test_corridor_size_and_feasibility(self):
        road_map = corridor_map()
        self.assertTrue(15 <= road_map.num_nodes() <= 25)
        self.assertTrue(road_map.chargers)
        self.assertTrue(is_feasible(road_map, "Chennai", "Bengaluru", 6, 10))

    def test_generated_maps_are_connected_with_quarter_chargers(self):
        for num_nodes in (20, 60, 150):
            generated = generate_map(num_nodes, 11)
            road_map = generated.road_map
            self.assertEqual(road_map.num_nodes(), num_nodes)
            self.assertEqual(len(road_map.chargers), round(num_nodes * 0.25))
            seen = {generated.start}
            queue = deque(seen)
            while queue:
                for neighbour in road_map.roads[queue.popleft()]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        queue.append(neighbour)
            self.assertEqual(len(seen), num_nodes)

    def test_generated_trip_is_feasible(self):
        for seed in range(20):
            g = generate_map(20, seed, battery_cap=4)
            self.assertTrue(is_feasible(g.road_map, g.start, g.destination, g.start_battery, 4))

    def test_generator_is_reproducible(self):
        self.assertEqual(fingerprint(generate_map(60, 5)), fingerprint(generate_map(60, 5)))

    def test_different_seeds_give_different_maps(self):
        self.assertNotEqual(fingerprint(generate_map(60, 5)), fingerprint(generate_map(60, 6)))

    def test_seeded_pairs_are_distinct_feasible_and_reproducible(self):
        road_map = generate_map(60, 5).road_map
        pairs = seeded_start_goal_pairs(road_map, 10, 1, 10, 5)
        self.assertEqual(len(set(pairs)), 10)
        self.assertEqual(pairs, seeded_start_goal_pairs(road_map, 10, 1, 10, 5))
        for start, goal in pairs:
            self.assertTrue(is_feasible(road_map, start, goal, 5, 10))


if __name__ == "__main__":
    unittest.main()
