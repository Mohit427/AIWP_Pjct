"""Unit tests for memory-bounded search (IDA*, DFBB), Random Walk, and route visualization."""

from __future__ import annotations

import os
import tempfile
import unittest

import matplotlib.pyplot as plt

from ev.charger_placement import (
    ChargerPlacementProblem,
    Trip,
    evaluate_charger_set,
)
from ev.informed import astar, ucs
from ev.local_search import steepest_ascent_hill_climbing
from ev.maps import MAP_SIZES, corridor_map, corridor_problem, generate_sized_map, toy_problem
from ev.memory_bounded import dfbb, dfbb_unbounded, ida_star, random_walk
from ev.metrics import SOLVED, measure
from ev.visualization import (
    format_placement_terminal,
    format_playback_terminal,
    format_route_terminal,
    reconstruct_route,
    render_placement_comparison,
    render_route_map,
    render_search_playback,
)


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
        self.assertIsNotNone(res_ida.peak_memory_kb)
        self.assertIsNotNone(res_dfbb.peak_memory_kb)

    def test_random_walk_baseline(self) -> None:
        """Random walk executes the requested steps, records trace history, and returns a valid state."""
        rm = corridor_map()
        candidates = [n for n in rm.coords if n not in rm.chargers][:8]
        trips = [Trip("Chennai", "Bengaluru", 6), Trip("Vellore", "Hosur", 5)]
        problem = ChargerPlacementProblem(
            road_map=rm,
            candidates=candidates,
            num_chargers=3,
            sample_trips=trips,
        )
        res = random_walk(problem, max_iterations=25, trace=True, seed=42)
        self.assertEqual(res.algorithm, "Random Walk")
        self.assertEqual(res.iterations, 25)
        self.assertTrue(problem.is_valid_state(res.best_state))
        self.assertEqual(len(res.trace_history), 26)
        self.assertAlmostEqual(
            res.best_value, evaluate_charger_set(problem, res.best_state), places=5
        )


class TestVisualization(unittest.TestCase):
    """Test route map renderer, search playback, and placement comparison views."""

    def test_route_map_and_renderers(self) -> None:
        """Verify reconstruct_route, terminal formatters, and matplotlib PNG rendering."""
        prob = corridor_problem()
        res = astar(prob)
        recon = reconstruct_route(prob, res)
        self.assertEqual(recon.steps[0].node, "Chennai")
        self.assertEqual(recon.steps[-1].node, "Bengaluru")
        self.assertEqual(recon.steps[-1].cumulative_minutes, res.total_minutes)

        term_route = format_route_terminal(prob, res)
        self.assertIn("Chennai -> Bengaluru", term_route)
        self.assertIn(str(res.total_minutes), term_route)

        term_play = format_playback_terminal(prob, res, max_steps=10)
        self.assertIn("SEARCH PLAYBACK", term_play)

        with tempfile.TemporaryDirectory() as tmpdir:
            route_png = os.path.join(tmpdir, "route_map.png")
            play_png = os.path.join(tmpdir, "playback.png")
            fig1 = render_route_map(prob, res, save_path=route_png)
            fig2 = render_search_playback(prob, res, num_snapshots=4, save_path=play_png)
            plt.close(fig1)
            plt.close(fig2)
            self.assertTrue(os.path.isfile(route_png))
            self.assertGreater(os.path.getsize(route_png), 1000)
            self.assertTrue(os.path.isfile(play_png))
            self.assertGreater(os.path.getsize(play_png), 1000)

    def test_placement_comparison_renderer(self) -> None:
        """Verify charger placement before-and-after terminal and matplotlib views."""
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