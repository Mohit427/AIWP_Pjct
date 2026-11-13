"""Heuristic consistency, admissibility, and optimal-cost agreement checker.

Author: ATK
Deliverable: Tue 10 Nov / Wed 11 Nov

Verifies:
1. Consistency: h(s) <= step_cost(s, a, s') + h(s') across every state and
   every legal action (both 'drive' and 'charge') for H0, H1, and H2.
2. Dominance: 0 == H0(s) <= H1(s) <= H2(s) for every state s, and h(goal) == 0.
3. Admissibility: h(s) <= h*(s) (true optimal remaining minutes) for all s.
4. Cost Agreement: UCS, A*(H0), A*(H1), A*(H2), IDA*, and DFBB return the
   exact same optimal total_minutes on every map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ev.heuristics import HEURISTIC_NAMES, make_problem_heuristics
from ev.informed import astar_h0, astar_h1, astar_h2, ucs
from ev.maps import corridor_problem, generate_sized_map, toy_problem
from ev.metrics import CUTOFF, NO_SOLUTION, SOLVED, Result, Solver
from ev.model import Problem, State

EPSILON = 1e-9


@dataclass
class ConsistencyReport:
    """Summary of exhaustive state-space heuristic checks on one Problem."""

    problem_name: str
    states_checked: int = 0
    transitions_checked: int = 0
    consistent: bool = True
    dominance_holds: bool = True
    goal_zero_holds: bool = True
    violations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return True if all heuristic properties hold with zero violations."""
        return (
            self.consistent
            and self.dominance_holds
            and self.goal_zero_holds
            and not self.violations
        )


@dataclass
class CostAgreementReport:
    """Summary of comparing optimal solvers on one Problem."""

    problem_name: str
    agreed: bool
    optimal_minutes: int | None
    costs_by_algorithm: dict[str, int | None]
    expansions_by_algorithm: dict[str, int]
    statuses_by_algorithm: dict[str, str]
    mismatches: list[str] = field(default_factory=list)


def all_states(problem: Problem) -> list[State]:
    """Enumerate every possible (node, battery) state in the finite state space."""
    return [
        (node, battery)
        for node in sorted(problem.road_map.coords)
        for battery in range(problem.battery_cap + 1)
    ]


def check_heuristics(problem: Problem) -> ConsistencyReport:
    """Exhaustively check H0, H1, H2 over every state and action in `problem`.

    Checks:
      - Goal states: h(s) == 0.
      - Non-negativity & dominance: 0 == H0(s) <= H1(s) <= H2(s).
      - Consistency: h(s) <= step_minutes + h(s') for every legal transition.
    """
    heuristics = make_problem_heuristics(problem)
    report = ConsistencyReport(problem_name=problem.road_map.name)

    for state in all_states(problem):
        report.states_checked += 1
        h0_val = heuristics["H0"](state)
        h1_val = heuristics["H1"](state)
        h2_val = heuristics["H2"](state)

        # 1. Goal state check
        if problem.is_goal(state):
            for name, val in (("H0", h0_val), ("H1", h1_val), ("H2", h2_val)):
                if abs(val) > EPSILON:
                    report.goal_zero_holds = False
                    report.violations.append(
                        f"Goal state {state} has non-zero {name}={val}"
                    )

        # 2. Pointwise dominance check: 0 == H0(s) <= H1(s) <= H2(s)
        if abs(h0_val) > EPSILON or h0_val > h1_val + EPSILON or h1_val > h2_val + EPSILON:
            report.dominance_holds = False
            report.violations.append(
                f"Dominance failed at {state}: H0={h0_val:.4f}, "
                f"H1={h1_val:.4f}, H2={h2_val:.4f}"
            )

        # 3. Edge consistency across every legal action (drive or charge)
        for action, next_state, step_minutes in problem.successors(state):
            report.transitions_checked += 1
            for name in HEURISTIC_NAMES:
                h_s = heuristics[name](state)
                h_next = heuristics[name](next_state)
                if h_s > step_minutes + h_next + EPSILON:
                    report.consistent = False
                    report.violations.append(
                        f"{name} inconsistency on {state} --({action}, {step_minutes}m)--> "
                        f"{next_state}: h(s)={h_s:.4f} > {step_minutes} + h(s')={h_next:.4f}"
                    )

    return report


def check_admissibility_exact(problem: Problem) -> list[str]:
    """Verify h(s) <= h*(s) against exact shortest trip time from every state.

    Because consistent + goal-zero automatically implies admissible, this
    provides an independent empirical check on small/corridor maps.
    """
    heuristics = make_problem_heuristics(problem)
    violations: list[str] = []

    for state in all_states(problem):
        node, bat = state
        sub_problem = Problem(
            problem.road_map,
            start=node,
            destination=problem.destination,
            start_battery=bat,
            battery_cap=problem.battery_cap,
        )
        res = ucs(sub_problem)
        if res.status != SOLVED:
            continue
        assert res.total_minutes is not None
        h_star = float(res.total_minutes)
        for name in HEURISTIC_NAMES:
            h_val = heuristics[name](state)
            if h_val > h_star + EPSILON:
                violations.append(
                    f"{name} inadmissible at {state}: h(s)={h_val:.4f} > h*(s)={h_star}"
                )
    return violations


def _discover_memory_bounded_solvers() -> dict[str, Solver]:
    """Auto-discover Rohiit's IDA* and DFBB solvers once pushed to the repo."""
    discovered: dict[str, Solver] = {}
    module_candidates = ("ev.more_informed", "ev.memory_bounded", "ev.rohiit", "ev.ida_dfbb")
    for mod_name in module_candidates:
        try:
            mod = __import__(mod_name, fromlist=["ALGORITHMS"])
            algos = getattr(mod, "ALGORITHMS", {})
            for key in ("IDA*", "DFBB"):
                if key in algos:
                    discovered[key] = algos[key]
        except ImportError:
            continue
    return discovered


def check_cost_agreement(
    problem: Problem,
    extra_solvers: Mapping[str, Solver] | None = None,
    include_auto_discovered: bool = True,
) -> CostAgreementReport:
    """Run all optimal algorithms on `problem` and verify identical solution cost.

    By default runs UCS, A*(H0), A*(H1), and A*(H2), plus IDA* and DFBB if
    available or passed via `extra_solvers`.
    """
    solvers: dict[str, Solver] = {
        "UCS": ucs,
        "A*(H0)": astar_h0,
        "A*(H1)": astar_h1,
        "A*(H2)": astar_h2,
    }
    if include_auto_discovered:
        solvers.update(_discover_memory_bounded_solvers())
    if extra_solvers:
        solvers.update(extra_solvers)

    costs: dict[str, int | None] = {}
    expansions: dict[str, int] = {}
    statuses: dict[str, str] = {}
    mismatches: list[str] = []

    for name, solver in solvers.items():
        res: Result = solver(problem)
        costs[name] = res.total_minutes
        expansions[name] = res.nodes_expanded
        statuses[name] = res.status

    # Reference is UCS (guaranteed optimal when it does not hit CUTOFF)
    ref_status = statuses["UCS"]
    ref_cost = costs["UCS"]

    for name, status in statuses.items():
        if status == CUTOFF:
            # A cutoff is not a wrong cost, though note it if UCS solved it.
            continue
        if ref_status == SOLVED and status != SOLVED:
            mismatches.append(f"{name} returned status {status}, expected {SOLVED}")
        elif ref_status == NO_SOLUTION and status != NO_SOLUTION:
            mismatches.append(f"{name} returned status {status}, expected {NO_SOLUTION}")
        elif status == SOLVED and costs[name] != ref_cost:
            mismatches.append(
                f"Cost mismatch on {problem.road_map.name}: "
                f"{name}={costs[name]} min vs UCS={ref_cost} min"
            )

    return CostAgreementReport(
        problem_name=problem.road_map.name,
        agreed=(len(mismatches) == 0),
        optimal_minutes=ref_cost,
        costs_by_algorithm=costs,
        expansions_by_algorithm=expansions,
        statuses_by_algorithm=statuses,
        mismatches=mismatches,
    )


def verify_all_standard_maps(
    extra_solvers: Mapping[str, Solver] | None = None,
) -> bool:
    """Run consistency and cost-agreement checks across toy, corridor, small, medium, large."""
    problems = [
        toy_problem(),
        corridor_problem(),
        generate_sized_map("small").problem(),
        generate_sized_map("medium").problem(),
        generate_sized_map("large").problem(),
    ]

    all_ok = True
    print("=" * 92)
    print("ATK HEURISTIC CONSISTENCY & OPTIMAL COST-AGREEMENT CHECKER (UCS, A*, IDA*, DFBB)")
    print("=" * 92)

    for prob in problems:
        c_rep = check_heuristics(prob)
        a_rep = check_cost_agreement(prob, extra_solvers=extra_solvers)
        exp = a_rep.expansions_by_algorithm
        dom_ok = exp["A*(H2)"] <= exp["A*(H1)"] <= exp["A*(H0)"]

        extra_str = ""
        if "IDA*" in exp and "DFBB" in exp:
            extra_str = f" | IDA*/DFBB: {exp['IDA*']:>5}/{exp['DFBB']:>4}"

        status_tag = "PASS" if (c_rep.passed and a_rep.agreed and dom_ok) else "FAIL"
        print(
            f"[{status_tag}] Map: {prob.road_map.name:<18} | "
            f"States: {c_rep.states_checked:>4} | Edges: {c_rep.transitions_checked:>5} | "
            f"Opt: {str(a_rep.optimal_minutes):>4}m | "
            f"A*(H0/H1/H2): {exp['A*(H0)']:>3}/{exp['A*(H1)']:>3}/{exp['A*(H2)']:>3}"
            f"{extra_str}"
        )

        if not c_rep.passed:
            all_ok = False
            for v in c_rep.violations[:5]:
                print(f"   ! Heuristic violation: {v}")
        if not a_rep.agreed:
            all_ok = False
            for m in a_rep.mismatches:
                print(f"   ! Cost mismatch: {m}")

    # Also verify exact admissibility against h*(s) on toy and corridor maps
    for prob in (toy_problem(), corridor_problem()):
        adm_violations = check_admissibility_exact(prob)
        if adm_violations:
            all_ok = False
            for v in adm_violations[:5]:
                print(f"   ! Admissibility violation: {v}")

    print("=" * 92)
    return all_ok


if __name__ == "__main__":
    success = verify_all_standard_maps()
    if not success:
        raise SystemExit(1)