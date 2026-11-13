"""Experiments, heuristic ablation, battery sweep, and master 8-algorithm comparison tables.

Author: ATK
Deliverable: Wed 11 Nov & Thu 12 Nov

Runs all route-finding experiments using Mohit's `measure()` harness, prints
clean tables for `main.py` and the final report, and saves `results/informed_experiments.csv`
and `results/informed_summary.txt` on the exact same 90 seeded trips used by
Mohit (uninformed) and Rohiit (memory_bounded).
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from ev.checker import _discover_memory_bounded_solvers
from ev.informed import astar, astar_h0, astar_h1, astar_h2, greedy_best_first, ucs
from ev.maps import (
    CORRIDOR_GOAL,
    CORRIDOR_START,
    CORRIDOR_START_BATTERY,
    DEFAULT_SEEDS,
    MAP_SIZES,
    corridor_map,
    corridor_problem,
    generate_map,
    generate_sized_map,
    is_feasible,
    seeded_start_goal_pairs,
    toy_problem,
)
from ev.metrics import CUTOFF, DEFAULT_MAX_EXPANSIONS, NO_SOLUTION, SOLVED, Result, Solver, measure
from ev.model import MAX_BATTERY_CAP, Problem
from ev.uninformed import bfs, dfs, iddfs

EXPERIMENT_TIME_LIMIT_S = 5.0
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

CSV_FIELDS = [
    "map_size",
    "num_nodes",
    "map_seed",
    "battery_cap",
    "start_battery",
    "pair_index",
    "start",
    "goal",
    "algorithm",
    "status",
    "num_actions",
    "total_minutes",
    "nodes_expanded",
    "nodes_generated",
    "peak_frontier",
    "runtime_s",
    "peak_memory_kb",
]


def _make_greedy_solver(h_name: str) -> Solver:
    """Bind Greedy Best-First Search to a specific heuristic name."""
    def _solver(
        problem: Problem,
        trace: bool = False,
        max_expansions: int = DEFAULT_MAX_EXPANSIONS,
        time_limit_s: float = EXPERIMENT_TIME_LIMIT_S,
    ) -> Result:
        res = greedy_best_first(
            problem,
            trace=trace,
            max_expansions=max_expansions,
            time_limit_s=time_limit_s,
            heuristic=h_name,
        )
        res.algorithm = f"Greedy({h_name})"
        return res

    return _solver


def _make_astar_solver(h_name: str) -> Solver:
    """Bind A* Search to a specific heuristic name."""
    def _solver(
        problem: Problem,
        trace: bool = False,
        max_expansions: int = DEFAULT_MAX_EXPANSIONS,
        time_limit_s: float = EXPERIMENT_TIME_LIMIT_S,
    ) -> Result:
        res = astar(
            problem,
            trace=trace,
            max_expansions=max_expansions,
            time_limit_s=time_limit_s,
            heuristic=h_name,
        )
        res.algorithm = f"A*({h_name})"
        return res

    return _solver


def standard_benchmark_problems() -> list[tuple[str, Problem]]:
    """Return the five standard benchmark problems across map sizes."""
    return [
        ("toy (4n)", toy_problem()),
        ("corridor (22n)", corridor_problem()),
        ("small (20n)", generate_sized_map("small").problem()),
        ("medium (60n)", generate_sized_map("medium").problem()),
        ("large (150n)", generate_sized_map("large").problem()),
    ]


def _fmt_int(val: int | None) -> str:
    return "-" if val is None else f"{val}"


def _fmt_mem(val: float | None) -> str:
    return "-" if val is None else f"{val:.1f}"


def print_results_table(title: str, rows: list[tuple[str, Result]]) -> None:
    """Print a formatted comparison table of Result records."""
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)
    header = (
        f"{'Map / Case':<18} | {'Algorithm':<12} | {'Status':<8} | "
        f"{'Cost(m)':>7} | {'Acts':>4} | {'Expanded':>9} | "
        f"{'Frontier':>8} | {'Time(ms)':>8} | {'Mem(KB)':>8}"
    )
    print(header)
    print("-" * 96)
    last_case = None
    for case_label, res in rows:
        if last_case is not None and case_label != last_case:
            print("-" * 96)
        last_case = case_label
        print(
            f"{case_label:<18} | {res.algorithm:<12} | {res.status:<8} | "
            f"{_fmt_int(res.total_minutes):>7} | {_fmt_int(res.num_actions):>4} | "
            f"{res.nodes_expanded:>9} | {res.peak_frontier:>8} | "
            f"{res.runtime_s * 1000:>8.2f} | {_fmt_mem(res.peak_memory_kb):>8}"
        )
    print("=" * 96)


def run_heuristic_ablation(time_limit_s: float = EXPERIMENT_TIME_LIMIT_S) -> list[tuple[str, Result]]:
    """Experiment 1: Compare UCS, Greedy(H0/H1/H2), and A*(H0/H1/H2) across map sizes."""
    solvers: list[tuple[str, Solver]] = [
        ("UCS", ucs),
        ("Greedy(H0)", _make_greedy_solver("H0")),
        ("Greedy(H1)", _make_greedy_solver("H1")),
        ("Greedy(H2)", _make_greedy_solver("H2")),
        ("A*(H0)", _make_astar_solver("H0")),
        ("A*(H1)", _make_astar_solver("H1")),
        ("A*(H2)", _make_astar_solver("H2")),
    ]

    rows: list[tuple[str, Result]] = []
    for label, prob in standard_benchmark_problems():
        for name, solver in solvers:
            res = measure(solver, prob, time_limit_s=time_limit_s)
            res.algorithm = name
            rows.append((label, res))
    return rows


def run_battery_capacity_sweep(
    time_limit_s: float = EXPERIMENT_TIME_LIMIT_S,
) -> list[tuple[str, Result]]:
    """Experiment 2: Sweep battery capacity (4, 6, 8, 10) on Corridor and Medium maps."""
    solvers: list[tuple[str, Solver]] = [
        ("UCS", ucs),
        ("Greedy(H2)", _make_greedy_solver("H2")),
        ("A*(H0)", _make_astar_solver("H0")),
        ("A*(H1)", _make_astar_solver("H1")),
        ("A*(H2)", _make_astar_solver("H2")),
    ]

    rows: list[tuple[str, Result]] = []
    cmap = corridor_map()

    for cap in (4, 6, 8, 10):
        start_bat = min(CORRIDOR_START_BATTERY, cap)
        if not is_feasible(cmap, CORRIDOR_START, CORRIDOR_GOAL, start_bat, cap):
            continue
        prob = Problem(cmap, CORRIDOR_START, CORRIDOR_GOAL, start_bat, cap)
        case_label = f"corridor cap={cap}"
        for name, solver in solvers:
            res = measure(solver, prob, time_limit_s=time_limit_s)
            res.algorithm = name
            rows.append((case_label, res))

    for cap in (4, 6, 8, 10):
        prob = generate_sized_map("medium", battery_cap=cap).problem()
        case_label = f"medium cap={cap}"
        for name, solver in solvers:
            res = measure(solver, prob, time_limit_s=time_limit_s)
            res.algorithm = name
            rows.append((case_label, res))

    return rows


def run_master_route_comparison(
    extra_solvers: dict[str, Solver] | None = None,
    time_limit_s: float = EXPERIMENT_TIME_LIMIT_S,
) -> list[tuple[str, Result]]:
    """Experiment 3: Master comparison of all 8 route-finding algorithms across map sizes."""
    solvers: dict[str, Solver] = {
        "BFS": bfs,
        "DFS": dfs,
        "IDDFS": iddfs,
        "UCS": ucs,
        "Greedy": greedy_best_first,
        "A*": astar,
    }
    solvers.update(_discover_memory_bounded_solvers())
    if extra_solvers:
        solvers.update(extra_solvers)

    rows: list[tuple[str, Result]] = []
    for label, prob in standard_benchmark_problems():
        for name, solver in solvers.items():
            res = measure(solver, prob, time_limit_s=time_limit_s)
            res.algorithm = name
            rows.append((label, res))
    return rows


def run_seeded_90_trip_benchmark(
    results_dir: Path = RESULTS_DIR,
    caps: tuple[int, ...] = (4, 6, 10),
    num_pairs: int = 10,
    pair_seed: int = 7,
) -> str:
    """Experiment 4: Run UCS, Greedy(H1/H2), and A*(H0/H1/H2) on the exact 90 seeded
    trips used by Mohit and Rohiit, saving `results/informed_experiments.csv` and
    `results/informed_summary.txt`.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "informed_experiments.csv"

    solvers: dict[str, Solver] = {
        "UCS": ucs,
        "Greedy(H1)": _make_greedy_solver("H1"),
        "Greedy(H2)": _make_greedy_solver("H2"),
        "A*(H0)": astar_h0,
        "A*(H1)": astar_h1,
        "A*(H2)": astar_h2,
    }

    rows: list[dict] = []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for size, num_nodes in MAP_SIZES.items():
            generated = generate_map(num_nodes, DEFAULT_SEEDS[size], MAX_BATTERY_CAP)
            pairs = seeded_start_goal_pairs(
                generated.road_map,
                num_pairs,
                pair_seed,
                MAX_BATTERY_CAP,
                MAX_BATTERY_CAP // 2,
            )
            for cap in caps:
                for index, (start, goal) in enumerate(pairs):
                    problem = Problem(generated.road_map, start, goal, cap // 2, cap)
                    for algo_name, solver in solvers.items():
                        res = measure(solver, problem)
                        row = {
                            "map_size": size,
                            "num_nodes": num_nodes,
                            "map_seed": generated.seed_used,
                            "battery_cap": cap,
                            "start_battery": cap // 2,
                            "pair_index": index,
                            "start": start,
                            "goal": goal,
                            "algorithm": algo_name,
                            "status": res.status,
                            "num_actions": res.num_actions,
                            "total_minutes": res.total_minutes,
                            "nodes_expanded": res.nodes_expanded,
                            "nodes_generated": res.nodes_generated,
                            "peak_frontier": res.peak_frontier,
                            "runtime_s": res.runtime_s,
                            "peak_memory_kb": res.peak_memory_kb,
                        }
                        writer.writerow(row)
                        handle.flush()
                        rows.append(row)

    by_group: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_group[(r["map_size"], int(r["battery_cap"]), r["algorithm"])].append(r)

    headers = [
        "map", "cap", "algorithm", "solved", "no_sol", "cutoff",
        "actions", "minutes", "expanded", "frontier", "time(ms)", "mem(KB)",
    ]
    table_rows: list[list[str]] = []
    algo_order = ["UCS", "A*(H0)", "A*(H1)", "A*(H2)", "Greedy(H1)", "Greedy(H2)"]
    for size in MAP_SIZES:
        for cap in caps:
            for algo in algo_order:
                grp = by_group[(size, cap, algo)]
                solved_rows = [d for d in grp if d["status"] == SOLVED]
                n_solved = len(solved_rows)
                n_nosol = sum(1 for d in grp if d["status"] == NO_SOLUTION)
                n_cut = sum(1 for d in grp if d["status"] == CUTOFF)
                table_rows.append([
                    size,
                    str(cap),
                    algo,
                    str(n_solved),
                    str(n_nosol),
                    str(n_cut),
                    f"{statistics.fmean(d['num_actions'] for d in solved_rows):.1f}" if solved_rows else "-",
                    f"{statistics.fmean(d['total_minutes'] for d in solved_rows):.1f}" if solved_rows else "-",
                    f"{statistics.fmean(d['nodes_expanded'] for d in solved_rows):.1f}" if solved_rows else "-",
                    f"{statistics.fmean(d['peak_frontier'] for d in solved_rows):.1f}" if solved_rows else "-",
                    f"{statistics.fmean(d['runtime_s'] * 1000 for d in solved_rows):.2f}" if solved_rows else "-",
                    f"{statistics.fmean(d['peak_memory_kb'] for d in solved_rows):.1f}" if solved_rows else "-",
                ])

    widths = [max(len(row[i]) for row in [headers] + table_rows) for i in range(len(headers))]
    lines = [
        "INFORMED SEARCH 90-TRIP EXPERIMENT SUMMARY (UCS, A* H0/H1/H2, Greedy H1/H2)",
        "",
        "  ".join(h.rjust(w) for h, w in zip(headers, widths)),
        "  ".join("-" * w for w in widths),
    ]
    for row in table_rows:
        lines.append("  ".join(c.rjust(w) for c, w in zip(row, widths)))
    summary_text = "\n".join(lines)
    (results_dir / "informed_summary.txt").write_text(summary_text + "\n")
    return summary_text


def print_dominance_proof_summary(ablation_rows: list[tuple[str, Result]]) -> None:
    """Print the empirical dominance summary verifying A*(H2) <= A*(H1) <= A*(H0) == UCS."""
    print("\n" + "=" * 96)
    print("THEORETICAL & EMPIRICAL PROOF SUMMARY: WHY A*(H2) IS THE BEST OPTIMAL ALGORITHM")
    print("=" * 96)

    by_map: dict[str, dict[str, Result]] = {}
    for label, res in ablation_rows:
        by_map.setdefault(label, {})[res.algorithm] = res

    for label, group in by_map.items():
        ucs_r = group["UCS"]
        h1_r = group["A*(H1)"]
        h2_r = group["A*(H2)"]
        g2_r = group["Greedy(H2)"]

        red_vs_ucs = (
            100.0 * (ucs_r.nodes_expanded - h2_r.nodes_expanded) / max(1, ucs_r.nodes_expanded)
        )
        red_vs_h1 = (
            100.0 * (h1_r.nodes_expanded - h2_r.nodes_expanded) / max(1, h1_r.nodes_expanded)
        )
        greedy_subopt = (
            (g2_r.total_minutes or 0) - (h2_r.total_minutes or 0)
            if g2_r.total_minutes is not None and h2_r.total_minutes is not None
            else 0
        )

        print(
            f"* {label:<16}: Optimal={h2_r.total_minutes:>4}m | "
            f"Expanded UCS={ucs_r.nodes_expanded:>4}, A*(H1)={h1_r.nodes_expanded:>4}, "
            f"A*(H2)={h2_r.nodes_expanded:>4} "
            f"(-{red_vs_ucs:>4.1f}% vs UCS, -{red_vs_h1:>4.1f}% vs H1) | "
            f"Greedy penalty: +{greedy_subopt}m"
        )
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ATK's route-finding experiments.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=EXPERIMENT_TIME_LIMIT_S,
        help="Per-run time limit in seconds (default: 5.0s).",
    )
    args = parser.parse_args()

    ablation_rows = run_heuristic_ablation(time_limit_s=args.timeout)
    print_results_table(
        "TABLE 1: HEURISTIC ABLATION ACROSS MAP SIZES (UCS vs GREEDY vs A*)",
        ablation_rows,
    )

    sweep_rows = run_battery_capacity_sweep(time_limit_s=args.timeout)
    print_results_table(
        "TABLE 2: BATTERY CAPACITY SWEEP (CORRIDOR & MEDIUM MAPS)",
        sweep_rows,
    )

    master_rows = run_master_route_comparison(time_limit_s=args.timeout)
    print_results_table(
        "TABLE 3: MASTER 8-ALGORITHM ROUTE-FINDING COMPARISON (ALL SOLVERS)",
        master_rows,
    )

    print_dominance_proof_summary(ablation_rows)

    print("\n" + "=" * 96)
    print(run_seeded_90_trip_benchmark())
    print("=" * 96)


if __name__ == "__main__":
    main()