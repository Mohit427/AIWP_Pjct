"""Tests for BFS, DFS, IDDFS and the measurement harness."""

import contextlib
import io
import random
import unittest

from ev.maps import corridor_problem, generate_map, toy_map, toy_problem
from ev.metrics import (
    CUTOFF,
    EXPANDED_ORDER_CAP,
    NO_SOLUTION,
    SOLVED,
    SearchRun,
    measure,
)
from ev.model import Action, Problem, RoadMap
from ev.uninformed import ALGORITHMS, bfs, dfs, iddfs

SOLVERS = list(ALGORITHMS.values())


def brute_force_min_actions(problem):
    """Return the fewest actions to the goal by trying every cycle-free path.

    Deliberately shares no code or strategy with BFS: plain recursion over all
    paths, keeping the shortest. Exponential, so only used on tiny maps.
    """
    best = None

    def visit(state, on_path, depth):
        nonlocal best
        if problem.is_goal(state):
            best = depth if best is None else min(best, depth)
            return
        for _action, next_state, _cost in problem.successors(state):
            if next_state not in on_path:
                on_path.add(next_state)
                visit(next_state, on_path, depth + 1)
                on_path.remove(next_state)

    start = problem.initial_state()
    visit(start, {start}, 0)
    return best


def line_map(battery_per_road=1, chargers=()):
    """Return the map P - Q - R, a straight line with optional chargers."""
    road_map = RoadMap("line")
    road_map.add_node("P", 0, 0)
    road_map.add_node("Q", 30, 0)
    road_map.add_node("R", 60, 0)
    road_map.add_road("P", "Q", 30, 30, battery_per_road)
    road_map.add_road("Q", "R", 30, 30, battery_per_road)
    for node in chargers:
        road_map.add_charger(node, 10, 20)
    return road_map


def random_trips(num_nodes, seeds):
    """Yield repeatable random trips, including ones with no solution."""
    for seed in seeds:
        generated = generate_map(num_nodes, seed)
        rng = random.Random(seed)
        names = sorted(generated.road_map.coords)
        for _ in range(3):
            start, goal = rng.sample(names, 2)
            cap = rng.choice([4, 6, 10])
            yield Problem(generated.road_map, start, goal, rng.randint(0, cap), cap)


def comparable(result):
    """Return a Result's fields without runtime, which varies between runs."""
    data = dict(vars(result))
    data.pop("runtime_s")
    return data


class EdgeCaseTests(unittest.TestCase):
    def test_start_equals_goal(self):
        problem = Problem(toy_map(), "S", "S", 1, 3)
        for solve in SOLVERS:
            result = solve(problem)
            self.assertEqual(result.status, SOLVED)
            self.assertEqual(result.path, [])
            self.assertEqual(result.total_minutes, 0)

    def test_goal_unreachable(self):
        road_map = line_map()
        road_map.add_node("Island", 500, 500)
        problem = Problem(road_map, "P", "Island", 3, 10)
        for solve in SOLVERS:
            result = solve(problem)
            self.assertEqual(result.status, NO_SOLUTION)
            self.assertFalse(result.found)
            self.assertIsNone(result.total_minutes)

    def test_battery_too_small_without_chargers(self):
        problem = Problem(line_map(), "P", "R", 1, 10)
        for solve in SOLVERS:
            self.assertEqual(solve(problem).status, NO_SOLUTION)

    def test_charging_rescues_a_small_battery(self):
        problem = Problem(line_map(chargers=["Q"]), "P", "R", 1, 10)
        for solve in SOLVERS:
            result = solve(problem)
            self.assertEqual(result.status, SOLVED)
            self.assertIn(Action("charge", "Q"), result.path)


class OptimalityTests(unittest.TestCase):
    def test_bfs_matches_independent_brute_force(self):
        checked = 0
        for seed in range(25):
            generated = generate_map(6, seed, battery_cap=3)
            names = sorted(generated.road_map.coords)
            for start in names[:3]:
                for goal in names[3:]:
                    problem = Problem(generated.road_map, start, goal, 2, 3)
                    expected = brute_force_min_actions(problem)
                    self.assertEqual(bfs(problem).num_actions, expected)
                    checked += expected is not None
        self.assertGreater(checked, 100)  # guards against testing only unsolvable trips

    def test_iddfs_matches_bfs_depth(self):
        for problem in random_trips(20, range(15)):
            breadth = bfs(problem)
            deepening = iddfs(problem, max_expansions=500_000, time_limit_s=20)
            if deepening.status != CUTOFF:
                self.assertEqual(deepening.status, breadth.status)
                self.assertEqual(deepening.num_actions, breadth.num_actions)

    def test_bfs_minimizes_actions_not_minutes(self):
        road_map = RoadMap("two-routes")
        road_map.add_node("P", 0, 0)
        road_map.add_node("M", 10, 0)
        road_map.add_node("R", 20, 0)
        road_map.add_road("P", "R", 20, 100, 1)  # one slow road
        road_map.add_road("P", "M", 10, 10, 1)  # two fast roads
        road_map.add_road("M", "R", 10, 10, 1)
        result = bfs(Problem(road_map, "P", "R", 5, 10))
        self.assertEqual(result.num_actions, 1)
        self.assertEqual(result.total_minutes, 100)


class CorrectnessTests(unittest.TestCase):
    def test_every_returned_path_is_valid(self):
        problems = list(random_trips(20, range(10))) + [corridor_problem(), toy_problem()]
        problems += [corridor_problem(shuffle_seed=seed) for seed in range(3)]
        for problem in problems:
            for solve in SOLVERS:
                result = solve(problem, max_expansions=300_000, time_limit_s=20)
                if result.status == SOLVED:
                    final_state, minutes = problem.replay(result.path)
                    self.assertTrue(problem.is_goal(final_state))
                    self.assertEqual(minutes, result.total_minutes)
                    self.assertEqual(len(result.path), result.num_actions)

    def test_dfs_agrees_with_bfs_on_solvability(self):
        for problem in random_trips(20, range(15)):
            self.assertEqual(dfs(problem).status, bfs(problem).status)

    def test_results_are_reproducible(self):
        for solve in SOLVERS:
            for seed in (None, 4):
                first = solve(corridor_problem(shuffle_seed=seed))
                second = solve(corridor_problem(shuffle_seed=seed))
                self.assertEqual(comparable(first), comparable(second))


class HarnessTests(unittest.TestCase):
    def test_expansion_limit_is_reported_as_cutoff(self):
        for solve in SOLVERS:
            result = solve(corridor_problem(), max_expansions=3)
            self.assertEqual(result.status, CUTOFF)
            self.assertFalse(result.found)
            self.assertLessEqual(result.nodes_expanded, 3)

    def test_time_limit_is_reported_as_cutoff(self):
        for solve in SOLVERS:
            self.assertEqual(solve(corridor_problem(), time_limit_s=0).status, CUTOFF)

    def test_trace_prints_one_line_per_expansion(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = bfs(toy_problem(), trace=True)
        self.assertEqual(output.getvalue().count("step"), result.nodes_expanded)

    def test_expanded_order_is_capped(self):
        run = SearchRun("x")
        for number in range(EXPANDED_ORDER_CAP + 10):
            run.record_expansion(("N", number), 0, 0)
        self.assertEqual(len(run.expanded_order), EXPANDED_ORDER_CAP)

    def test_measure_fills_memory_and_keeps_counts(self):
        for solve in SOLVERS:
            measured = measure(solve, toy_problem())
            self.assertGreater(measured.peak_memory_kb, 0)
            self.assertEqual(measured.nodes_expanded, solve(toy_problem()).nodes_expanded)

    def test_solved_rejects_a_path_that_misses_the_goal(self):
        with self.assertRaises(RuntimeError):
            SearchRun("x").solved(toy_problem(), [Action("drive", "A")])


if __name__ == "__main__":
    unittest.main()
