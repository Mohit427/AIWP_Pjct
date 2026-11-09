"""Unit tests for ATK's heuristics and informed search algorithms (UCS, Greedy, A*)."""

from __future__ import annotations

import unittest

from ev.heuristics import HEURISTIC_NAMES, make_problem_heuristics
from ev.informed import astar, greedy_best_first, ucs
from ev.maps import corridor_problem, generate_sized_map, toy_problem
from ev.metrics import SOLVED, measure
from ev.checker import (
    check_admissibility_exact,
    check_cost_agreement,
    check_heuristics,
)
from ev.maps import MAP_SIZES


class TestInformedSearch(unittest.TestCase):
    """Verify UCS, Greedy, A*, and heuristics H0, H1, H2."""

    def test_toy_problem_optimality_and_trap_avoidance(self) -> None:
        """On toy_map with start_battery=2, B strands the car; S->A(charge)->G costs 70 min."""
        problem = toy_problem(start_battery=2, battery_cap=3)

        res_ucs = ucs(problem)
        res_greedy = greedy_best_first(problem)
        res_astar = astar(problem)

        self.assertEqual(res_ucs.status, SOLVED)
        self.assertEqual(res_greedy.status, SOLVED)
        self.assertEqual(res_astar.status, SOLVED)

        # S->A (30m, 1u) + Charge A (10m, +1u) + A->G (30m, 2u) = 70 minutes
        self.assertEqual(res_ucs.total_minutes, 70)
        self.assertEqual(res_astar.total_minutes, 70)

    def test_corridor_cost_agreement_and_dominance(self) -> None:
        """UCS and A* (with H0, H1, H2) must return identical optimal cost,
        and H2 must expand no more nodes than H1 or H0.
        """
        problem = corridor_problem()
        res_ucs = measure(ucs, problem)
        res_h0 = astar(problem, heuristic="H0")
        res_h1 = astar(problem, heuristic="H1")
        res_h2 = measure(astar, problem)

        self.assertEqual(res_ucs.status, SOLVED)
        for res in (res_h0, res_h1, res_h2):
            self.assertEqual(res.status, SOLVED)
            self.assertEqual(res.total_minutes, res_ucs.total_minutes)

        # Heuristic dominance: nodes expanded should satisfy H2 <= H1 <= H0
        self.assertLessEqual(res_h2.nodes_expanded, res_h1.nodes_expanded)
        self.assertLessEqual(res_h1.nodes_expanded, res_h0.nodes_expanded)
        self.assertIsNotNone(res_h2.peak_memory_kb)

    def test_heuristic_dominance_and_goal_zero(self) -> None:
        """Verify 0 == H0(s) <= H1(s) <= H2(s) and H(goal) == 0."""
        problem = generate_sized_map("small").problem()
        hs = make_problem_heuristics(problem)

        for node in problem.road_map.coords:
            for bat in range(problem.battery_cap + 1):
                state = (node, bat)
                v0, v1, v2 = (hs[k](state) for k in HEURISTIC_NAMES)
                if problem.is_goal(state):
                    self.assertEqual((v0, v1, v2), (0.0, 0.0, 0.0))
                else:
                    self.assertEqual(v0, 0.0)
                    self.assertLessEqual(v0, v1 + 1e-9)
                    self.assertLessEqual(v1, v2 + 1e-9)

class TestChecker(unittest.TestCase):
    """Verify ATK's consistency, admissibility, and cost-agreement checker."""

    def test_consistency_across_all_maps(self) -> None:
        """H0, H1, H2 must be consistent across every state and edge on all maps."""
        problems = [toy_problem(), corridor_problem()] + [
            generate_sized_map(size).problem() for size in MAP_SIZES
        ]
        for prob in problems:
            report = check_heuristics(prob)
            self.assertTrue(
                report.passed,
                msg=f"Consistency failed on {prob.road_map.name}: {report.violations[:3]}",
            )

    def test_exact_admissibility_on_toy_and_corridor(self) -> None:
        """H0, H1, H2 must never exceed true optimal remaining cost h*(s)."""
        for prob in (toy_problem(), corridor_problem()):
            violations = check_admissibility_exact(prob)
            self.assertEqual(violations, [])

    def test_cost_agreement_across_all_maps(self) -> None:
        """UCS, A*(H0), A*(H1), and A*(H2) must return identical optimal minutes."""
        problems = [toy_problem(), corridor_problem()] + [
            generate_sized_map(size).problem() for size in MAP_SIZES
        ]
        for prob in problems:
            report = check_cost_agreement(prob)
            self.assertTrue(
                report.agreed,
                msg=f"Cost disagreement on {prob.road_map.name}: {report.mismatches}",
            )


if __name__ == "__main__":
    unittest.main()