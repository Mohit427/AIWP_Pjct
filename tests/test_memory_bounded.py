"""Unit tests, edge-case tests, visualization tests, and main.py integration tests."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest

try:
    import matplotlib.pyplot as plt
    from ev.viz import (
        format_placement_terminal,
        format_playback_terminal,
        format_route_terminal,
        reconstruct_route,
        render_memory_comparison,
        render_placement_comparison,
        visualize,
    )
    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False

from ev.charger_placement import (
    ChargerPlacementProblem,
    Trip,
    evaluate_charger_set,
)
from ev.informed import astar, ucs
from ev.local_search import steepest_ascent_hill_climbing
from ev.maps import MAP_SIZES, corridor_map, corridor_problem, generate_sized_map, toy_map, toy_problem
from ev.memory_bounded import dfbb, dfbb_unbounded, ida_star, random_walk
from ev.metrics import CUTOFF, NO_SOLUTION, SOLVED, measure
from ev.model import Problem, RoadMap
from ev.uninformed import bfs, dfs, iddfs
import main as main_mod


class TestMemoryBoundedAndRandomWalk(unittest.TestCase):
    """Test IDA*, DFBB (with and without starting bound), and Random Walk."""

    def test_ida_star_and_dfbb_optimal_on_all_map_sizes(self) -> None:
        """IDA* and DFBB must match UCS/A* optimal cost on toy, corridor, small, medium, large."""
        problems = [toy_problem(), corridor_problem()] + [
            generate_sized_map(size).problem() for size in MAP_SIZES
        ]
        for prob in problems:
            res_ucs = ucs(prob)
            res_ida = ida_star(prob)
            res_dfbb = dfbb(prob)

            self.assertEqual(res_ida.status, SOLVED)
            self.assertEqual(res_dfbb.status, SOLVED)
            self.assertEqual(res_ida.total_minutes, res_ucs.total_minutes)
            self.assertEqual(res_dfbb.total_minutes, res_ucs.total_minutes)

    def test_dfbb_with_and_without_starting_bound(self) -> None:
        """DFBB must return the same optimal cost with or without a good starting bound,
        and a tight starting bound should expand no more nodes than an unbounded run.
        """
        prob = generate_sized_map("medium").problem()
        res_bounded = dfbb(prob, use_greedy_bound=True)
        res_unbounded = dfbb_unbounded(prob)
        res_tight_explicit = dfbb(
            prob, initial_bound=res_bounded.total_minutes, use_greedy_bound=False
        )

        self.assertEqual(res_bounded.status, SOLVED)
        self.assertEqual(res_unbounded.status, SOLVED)
        self.assertEqual(res_tight_explicit.status, SOLVED)
        self.assertEqual(res_bounded.total_minutes, res_unbounded.total_minutes)
        self.assertEqual(res_tight_explicit.total_minutes, res_unbounded.total_minutes)
        self.assertLessEqual(res_bounded.nodes_expanded, res_unbounded.nodes_expanded)

    def test_memory_bounded_lower_peak_frontier_than_astar(self) -> None:
        """IDA* and DFBB maintain O(b*d) active frontier on corridor map."""
        prob = corridor_problem()
        res_ast = measure(astar, prob)
        res_ida = measure(ida_star, prob)
        res_dfbb = measure(dfbb, prob)

        self.assertLess(res_ida.peak_frontier, res_ast.peak_frontier)
        self.assertLess(res_dfbb.peak_frontier, res_ast.peak_frontier)
        self.assertIsNotNone(res_ida.peak_memory_kb)
        self.assertIsNotNone(res_dfbb.peak_memory_kb)

    def test_random_walk_placement_and_route(self) -> None:
        """Random walk supports both ChargerPlacementProblem and route-finding Problem."""
        rm = corridor_map()
        candidates = [n for n in rm.coords if n not in rm.chargers][:8]
        trips = [Trip("Chennai", "Bengaluru", 6), Trip("Vellore", "Hosur", 5)]
        placement_prob = ChargerPlacementProblem(
            road_map=rm,
            candidates=candidates,
            num_chargers=3,
            sample_trips=trips,
        )
        res = random_walk(placement_prob, max_iterations=25, trace=True, seed=42)
        self.assertEqual(res.algorithm, "Random Walk")
        self.assertEqual(res.iterations, 25)
        self.assertTrue(placement_prob.is_valid_state(res.best_state))
        self.assertEqual(len(res.trace_history), 26)
        self.assertAlmostEqual(
            res.best_value, evaluate_charger_set(placement_prob, res.best_state), places=5
        )

        route_res = measure(random_walk, toy_problem())
        self.assertIn(route_res.status, (SOLVED, CUTOFF))
        self.assertIsNotNone(route_res.peak_memory_kb)

    def test_contract_trace_and_limits(self) -> None:
        """Verify CONTRACT.md rules: trace prints one 'step' per expansion, limits return CUTOFF."""
        for solver in (ida_star, dfbb):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                res = solver(toy_problem(), trace=True)
            self.assertEqual(output.getvalue().count("step"), res.nodes_expanded)

            cut = solver(corridor_problem(), max_expansions=3)
            self.assertEqual(cut.status, CUTOFF)
            self.assertLessEqual(cut.nodes_expanded, 3)


class TestEdgeCases(unittest.TestCase):
    """Wednesday (11 Nov) edge-case tests:
      1. Start equals goal
      2. Goal unreachable
      3. Battery too small
      4. No chargers on the route
    """

    def test_start_equals_goal(self) -> None:
        """When start == destination, solvers return SOLVED with 0 actions and 0 minutes."""
        prob = Problem(toy_map(), "S", "S", start_battery=2, battery_cap=3)
        for solver in (bfs, dfs, iddfs, ucs, astar, ida_star, dfbb):
            res = solver(prob)
            self.assertEqual(res.status, SOLVED)
            self.assertEqual(res.path, [])
            self.assertEqual(res.total_minutes, 0)
            self.assertEqual(res.num_actions, 0)

    def test_goal_unreachable(self) -> None:
        """When destination is in a disconnected graph component, solvers return NO_SOLUTION."""
        rm = RoadMap("disconnected")
        rm.add_node("A", 0, 0)
        rm.add_node("B", 20, 0)
        rm.add_node("Island", 200, 200)
        rm.add_road("A", "B", 25, 25, 1)
        prob = Problem(rm, "A", "Island", start_battery=5, battery_cap=10)
        for solver in (bfs, dfs, iddfs, ucs, astar, ida_star, dfbb):
            res = solver(prob)
            self.assertEqual(res.status, NO_SOLUTION)
            self.assertFalse(res.found)
            self.assertIsNone(res.total_minutes)

    def test_battery_too_small(self) -> None:
        """When battery cannot cover the very first road and no charger is at start."""
        rm = RoadMap("tight_battery")
        rm.add_node("Start", 0, 0)
        rm.add_node("Goal", 80, 0)
        rm.add_road("Start", "Goal", 90, 60, 3)
        prob = Problem(rm, "Start", "Goal", start_battery=2, battery_cap=2)
        for solver in (bfs, dfs, iddfs, ucs, astar, ida_star, dfbb):
            res = solver(prob)
            self.assertEqual(res.status, NO_SOLUTION)
            self.assertFalse(res.found)

    def test_no_chargers_on_route(self) -> None:
        """When all chargers on the corridor are removed and starting battery cannot reach goal."""
        rm = corridor_map()
        rm.chargers.clear()
        prob = Problem(rm, "Chennai", "Bengaluru", start_battery=4, battery_cap=10)
        for solver in (bfs, dfs, iddfs, ucs, astar, ida_star, dfbb):
            res = solver(prob)
            self.assertEqual(res.status, NO_SOLUTION)
            self.assertFalse(res.found)


class TestVisualizationAndMainIntegration(unittest.TestCase):
    """Test route map, playback, placement comparison, memory comparison, and main.py wiring."""

    def test_main_available_algorithms(self) -> None:
        """Verify main.py discovers all 8 route-finding algorithms."""
        algos = main_mod.available_algorithms()
        for name in ("BFS", "DFS", "IDDFS", "UCS", "Greedy", "A*", "IDA*", "DFBB"):
            self.assertIn(name, algos)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
    def test_route_map_playback_and_memory_renderers(self) -> None:
        """Verify reconstruct_route, terminal formatters, and matplotlib PNG rendering."""
        prob = corridor_problem()
        res_ast = measure(astar, prob)
        res_ida = measure(ida_star, prob)
        res_dfbb = measure(dfbb, prob)
        recon = reconstruct_route(prob, res_ast)
        self.assertEqual(recon.steps[0].node, "Chennai")
        self.assertEqual(recon.steps[-1].node, "Bengaluru")
        self.assertEqual(recon.steps[-1].cumulative_minutes, res_ast.total_minutes)

        term_route = format_route_terminal(prob, res_ast)
        self.assertIn("Chennai -> Bengaluru", term_route)
        self.assertIn(str(res_ast.total_minutes), term_route)

        term_play = format_playback_terminal(prob, res_ast, max_steps=10)
        self.assertIn("SEARCH PLAYBACK", term_play)

        with tempfile.TemporaryDirectory() as tmpdir:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                saved = visualize(
                    prob,
                    {"A*": res_ast, "IDA*": res_ida, "DFBB": res_dfbb},
                    save_dir=tmpdir,
                )
            self.assertTrue(saved["route_map"].is_file())
            self.assertTrue(saved["search_playback"].is_file())

            mem_png = os.path.join(tmpdir, "memory_cmp.png")
            fig_m = render_memory_comparison(
                {"corridor": {"A*": res_ast, "IDA*": res_ida, "DFBB": res_dfbb}},
                save_path=mem_png,
            )
            plt.close(fig_m)
            self.assertTrue(os.path.isfile(mem_png))
            self.assertGreater(os.path.getsize(mem_png), 1000)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not installed")
    def test_placement_comparison_and_main_integration(self) -> None:
        """Verify charger placement comparison renderer."""
        rm = corridor_map()
        candidates = [n for n in rm.coords if n not in rm.chargers][:8]
        trips = [Trip("Chennai", "Bengaluru", 6), Trip("Arakkonam", "Hosur", 5)]
        problem = ChargerPlacementProblem(
            road_map=rm,
            candidates=candidates,
            num_chargers=3,
            sample_trips=trips,
        )
        init_state = problem.initial_state(seed=7)
        hc_res = steepest_ascent_hill_climbing(
            problem, initial_state=init_state, trace=True, seed=7
        )
        term_out = format_placement_terminal(problem, init_state, hc_res)
        self.assertIn("CHARGER PLACEMENT BEFORE & AFTER", term_out)

        with tempfile.TemporaryDirectory() as tmpdir:
            place_png = os.path.join(tmpdir, "placement.png")
            fig = render_placement_comparison(
                problem, init_state, hc_res, save_path=place_png
            )
            plt.close(fig)
            self.assertTrue(os.path.isfile(place_png))
            self.assertGreater(os.path.getsize(place_png), 1000)


if __name__ == "__main__":
    unittest.main()