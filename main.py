"""Command-line entry point for the EV trip planner.

    python main.py                 interactive menu
    python main.py --demo          non-interactive tour of the features
    python main.py --experiments   run the experiments and save to results/
"""

from __future__ import annotations

import csv
import importlib
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from ev.maps import (
    DEFAULT_SEEDS,
    MAP_SIZES,
    corridor_problem,
    generate_map,
    generate_sized_map,
    seeded_start_goal_pairs,
    toy_problem,
)
from ev.metrics import CUTOFF, NO_SOLUTION, SOLVED, Result, measure
from ev.model import CHARGE, MAX_BATTERY_CAP, Problem
from ev.uninformed import ALGORITHMS as UNINFORMED_ALGORITHMS

OPTIONAL_ALGORITHM_MODULES = ["ev.informed", "ev.local_search", "ev.more_informed"]
RESULTS_DIR = Path(__file__).parent / "results"

EXPERIMENT_CAPS = (4, 6, 10)
EXPERIMENT_PAIRS = 10
EXPERIMENT_PAIR_SEED = 7
EXPERIMENT_ALGORITHMS = ("BFS", "DFS", "IDDFS")

# Smaller than the solver defaults, so the menu and the demo stay responsive.
# Any run stopped by these limits is shown as a cutoff.
INTERACTIVE_MAX_EXPANSIONS = 300_000
INTERACTIVE_TIME_LIMIT_S = 10
DEFAULT_TRACE_EXPANSIONS = 200

MAP_CHOICES = ("toy", "corridor", "small", "medium", "large")
VERDICT_SCENARIOS = ("corridor", "small", "medium", "large")


def _optional_module(name: str):
    """Import a teammate's module, or return None if it does not exist yet.

    Only a missing module is treated as normal. If the module exists but one
    of its own imports fails, a warning is printed so the bug is not hidden.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            print(f"WARNING: {name} failed to import: {error}")
        return None
    except Exception as error:  # a teammate's bug must not stop the whole program
        print(f"WARNING: {name} failed to import: {error}")
        return None


def available_algorithms() -> dict:
    """Return every algorithm present, as a dictionary of name to solver."""
    algorithms = dict(UNINFORMED_ALGORITHMS)
    for module_name in OPTIONAL_ALGORITHM_MODULES:
        module = _optional_module(module_name)
        if module is None:
            continue
        extra = getattr(module, "ALGORITHMS", None)
        if isinstance(extra, dict):
            algorithms.update(extra)
        else:
            print(f"WARNING: {module_name} has no ALGORITHMS dictionary (see CONTRACT.md)")
    return algorithms


def build_problem(choice: str) -> Problem:
    """Return the standard trip for a map name in MAP_CHOICES."""
    if choice == "toy":
        return toy_problem()
    if choice == "corridor":
        return corridor_problem()
    return generate_sized_map(choice).problem()


def charging_cost(problem: Problem, result: Result) -> int | None:
    """Return the money spent on charging along a solved path, in rupees.

    Minutes are the search cost. Price is a second measure of the same plan,
    calculated afterwards. None of my algorithms optimize it.
    """
    if not result.found:
        return None
    chargers = problem.road_map.chargers
    return sum(chargers[a.target].price_per_unit for a in result.path if a.kind == CHARGE)


def print_table(headers: list[str], rows: list[list]) -> str:
    """Print rows as an aligned table and return the same text.

    Missing values (None) appear as "-" rather than 0, because a zero would
    look like a real measurement.
    """
    cells = [["-" if value is None else str(value) for value in row] for row in rows]
    widths = [max(len(row[i]) for row in [headers] + cells) for i in range(len(headers))]
    lines = ["  ".join(text.rjust(width) for text, width in zip(headers, widths))]
    lines.append("  ".join("-" * width for width in widths))
    for row in cells:
        lines.append("  ".join(text.rjust(width) for text, width in zip(row, widths)))
    text = "\n".join(lines)
    print(text)
    return text


def show_map(problem: Problem) -> None:
    """Print a map's trip and size. Small maps also list every road and charger."""
    road_map = problem.road_map
    print(f"\nMap: {road_map.name}")
    print(
        f"Nodes: {road_map.num_nodes()}  Roads: {road_map.num_roads()}  "
        f"Chargers: {len(road_map.chargers)}"
    )
    print(
        f"Trip: {problem.start} -> {problem.destination}  "
        f"start battery {problem.start_battery}, cap {problem.battery_cap}"
    )
    if road_map.num_nodes() > 25:
        print("(road list omitted: more than 25 nodes)")
        return
    rows = [
        [a, b, road.km, road.minutes, road.battery_cost]
        for a, links in sorted(road_map.roads.items())
        for b, road in sorted(links.items())
        if a < b
    ]
    print_table(["from", "to", "km", "min", "battery"], rows)
    print()
    chargers = sorted(road_map.chargers.items())
    print_table(
        ["charger", "min/unit", "price/unit"],
        [[node, c.minutes_per_unit, c.price_per_unit] for node, c in chargers],
    )


def print_result(problem: Problem, result: Result) -> None:
    """Print one algorithm's outcome and, if solved, its plan."""
    print(
        f"\n{result.algorithm}: {result.status}, actions={result.num_actions}, "
        f"minutes={result.total_minutes}, charging cost={charging_cost(problem, result)}, "
        f"expanded={result.nodes_expanded}"
    )
    if result.found:
        print("Plan: " + (", ".join(str(a) for a in result.path) or "(already there)"))


def compare(
    problem: Problem,
    algorithms: dict,
    max_expansions: int = INTERACTIVE_MAX_EXPANSIONS,
    time_limit_s: float = INTERACTIVE_TIME_LIMIT_S,
) -> dict[str, Result]:
    """Measure each algorithm on one problem and return results by name.

    A crash in one algorithm is reported and the others still run, so one
    unfinished module cannot stop a team comparison.
    """
    results: dict[str, Result] = {}
    for name, solver in algorithms.items():
        try:
            results[name] = measure(solver, problem, max_expansions, time_limit_s)
        except Exception as error:
            print(f"WARNING: {name} crashed: {error!r}")
    return results


def print_comparison(problem: Problem, results: dict[str, Result]) -> None:
    """Print the comparison table for one problem."""
    rows = [
        [
            name,
            r.status,
            r.num_actions,
            charging_cost(problem, r),
            r.total_minutes,
            r.nodes_expanded,
            r.peak_frontier,
            f"{r.runtime_s * 1000:.2f}",
            f"{r.peak_memory_kb:.1f}",
        ]
        for name, r in results.items()
    ]
    print()
    print_table(
        ["algorithm", "status", "actions", "cost(Rs)", "minutes", "expanded",
         "peak frontier", "time(ms)", "memory(KB)"],
        rows,
    )


VERDICT_CRITERIA = (
    ("fewest minutes", lambda r: r.total_minutes),
    ("fewest actions", lambda r: r.num_actions),
    ("fewest expanded", lambda r: r.nodes_expanded),
    ("lowest memory", lambda r: r.peak_memory_kb),
    ("fastest", lambda r: r.runtime_s),
)


def winners(results: dict[str, Result], key) -> str:
    """Return the best solved algorithm(s) under a measure, listing exact ties.

    Only solved runs qualify: a cutoff did little work because it gave up, so
    letting it win "fewest expanded" would reward failure.
    """
    solved = {name: key(r) for name, r in results.items() if r.status == SOLVED}
    if not solved:
        return "none solved"
    best = min(solved.values())
    return "+".join(name for name, value in solved.items() if value == best)


def verdict(algorithms: dict) -> None:
    """Print the winner per sub-problem among the algorithms that exist so far."""
    print("\n" + "=" * 70)
    print(f"VERDICT among available algorithms: {', '.join(algorithms)}")
    print(
        f"Limits: {INTERACTIVE_MAX_EXPANSIONS:,} expansions, "
        f"{INTERACTIVE_TIME_LIMIT_S} s. One run per map; time is noisy."
    )
    print("=" * 70)
    rows = []
    unsolved = []
    for scenario in VERDICT_SCENARIOS:
        results = compare(build_problem(scenario), algorithms)
        rows.append([scenario] + [winners(results, key) for _label, key in VERDICT_CRITERIA])
        unsolved += [f"{scenario}/{n}: {r.status}" for n, r in results.items() if r.status != SOLVED]
    print_table(["sub-problem"] + [label for label, _key in VERDICT_CRITERIA], rows)
    print("Not eligible to win: " + ("; ".join(unsolved) if unsolved else "none"))
    print("This is provisional until every team algorithm is present.")


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

CSV_FIELDS = [
    "map_size", "num_nodes", "map_seed", "battery_cap", "start_battery", "pair_index",
    "start", "goal", "algorithm", "status", "num_actions", "total_minutes",
    "nodes_expanded", "nodes_generated", "peak_frontier", "runtime_s", "peak_memory_kb",
]


def run_experiments(results_dir: Path = RESULTS_DIR) -> None:
    """Run BFS, DFS and IDDFS over sizes x caps x start-goal pairs and save the data.

    One fixed map per size, and the same trips at every cap, so only the cap
    changes between cap comparisons. Each trip starts at half the cap. A trip
    verified at cap 10 may be impossible at a smaller cap; that is recorded as
    no_solution and is a real result. Rows are written as they finish so an
    interrupted run keeps its data.
    """
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "experiments.csv"
    rows: list[dict] = []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for size, num_nodes in MAP_SIZES.items():
            generated = generate_map(num_nodes, DEFAULT_SEEDS[size], MAX_BATTERY_CAP)
            pairs = seeded_start_goal_pairs(
                generated.road_map, EXPERIMENT_PAIRS, EXPERIMENT_PAIR_SEED,
                MAX_BATTERY_CAP, MAX_BATTERY_CAP // 2,
            )
            for cap in EXPERIMENT_CAPS:
                print(f"running {size} (seed {generated.seed_used}), cap {cap} ...", flush=True)
                for index, (start, goal) in enumerate(pairs):
                    problem = Problem(generated.road_map, start, goal, cap // 2, cap)
                    for name in EXPERIMENT_ALGORITHMS:
                        result = measure(UNINFORMED_ALGORITHMS[name], problem)
                        row = {
                            "map_size": size, "num_nodes": num_nodes,
                            "map_seed": generated.seed_used, "battery_cap": cap,
                            "start_battery": cap // 2, "pair_index": index,
                            "start": start, "goal": goal, "algorithm": name,
                            "status": result.status, "num_actions": result.num_actions,
                            "total_minutes": result.total_minutes,
                            "nodes_expanded": result.nodes_expanded,
                            "nodes_generated": result.nodes_generated,
                            "peak_frontier": result.peak_frontier,
                            "runtime_s": result.runtime_s,
                            "peak_memory_kb": result.peak_memory_kb,
                        }
                        writer.writerow(row)
                        handle.flush()
                        rows.append(row)
    text = summarize(rows)
    print("\n" + text)
    (results_dir / "summary.txt").write_text(text + "\n")
    print(f"\nSaved {csv_path} and {results_dir / 'summary.txt'}")


def _mean(values: list[float]) -> float | None:
    """Return the average, or None for an empty list."""
    return statistics.fmean(values) if values else None


def _format(value: float | None, pattern: str = "{:.1f}") -> str | None:
    """Format a number, passing None through so the table can show '-'."""
    return None if value is None else pattern.format(value)


def summarize(rows: list[dict]) -> str:
    """Build the summary text from experiment rows.

    Averages use only "common" trips, those solved by all three algorithms at
    that setting. Otherwise an algorithm that gave up on the hard trips would
    be averaged over easy trips only and look better than it is. Status counts
    cover every trip, so cutoffs always remain visible.
    """
    by_trip: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_trip[(row["map_size"], row["battery_cap"], row["pair_index"])][row["algorithm"]] = row

    table_rows = []
    for size in MAP_SIZES:
        for cap in EXPERIMENT_CAPS:
            trips = [t for key, t in by_trip.items() if key[0] == size and key[1] == cap]
            common = [t for t in trips if all(r["status"] == SOLVED for r in t.values())]
            for name in EXPERIMENT_ALGORITHMS:
                statuses = [t[name]["status"] for t in trips]
                data = [t[name] for t in common]
                table_rows.append([
                    size, cap, name,
                    statuses.count(SOLVED), statuses.count(NO_SOLUTION), statuses.count(CUTOFF),
                    len(common),
                    _format(_mean([d["num_actions"] for d in data])),
                    _format(_mean([d["total_minutes"] for d in data])),
                    _format(_mean([d["nodes_expanded"] for d in data])),
                    _format(_mean([d["peak_frontier"] for d in data])),
                    _format(_mean([d["runtime_s"] * 1000 for d in data]), "{:.2f}"),
                    _format(_mean([d["peak_memory_kb"] for d in data])),
                ])

    table = print_table(
        ["map", "cap", "algorithm", "solved", "no_sol", "cutoff", "common", "actions",
         "minutes", "expanded", "frontier", "time(ms)", "mem(KB)"],
        table_rows,
    )
    both = [t for t in by_trip.values() if t["BFS"]["status"] == SOLVED and t["IDDFS"]["status"] == SOLVED]
    same_depth = sum(t["BFS"]["num_actions"] == t["IDDFS"]["num_actions"] for t in both)
    cutoffs = [r for r in rows if r["status"] == CUTOFF]
    lines = [
        "EXPERIMENT SUMMARY (generated from results/experiments.csv)",
        "Averages cover only 'common' trips: solved by all three at that setting.",
        "",
        table,
        "",
        f"Total runs: {len(rows)}. Cutoffs: {len(cutoffs)}.",
        f"BFS and IDDFS both solved {len(both)} trips; same action count on {same_depth}.",
    ]
    for name in EXPERIMENT_ALGORITHMS:
        count = sum(1 for r in cutoffs if r["algorithm"] == name)
        lines.append(f"{name} cutoffs: {count}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# User interface
# --------------------------------------------------------------------------


def _ask_choice(prompt: str, options: list[str]) -> str:
    """Show numbered options and return the one chosen."""
    for number, option in enumerate(options, 1):
        print(f"  {number}. {option}")
    while True:
        answer = input(f"{prompt}: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        print("Please enter one of the numbers shown.")


def _run_with_trace(algorithms: dict) -> None:
    """Menu action: run one algorithm and print one line per expansion."""
    name = _ask_choice("Algorithm", list(algorithms))
    problem = build_problem(_ask_choice("Map", list(MAP_CHOICES)))
    answer = input(f"Max expansions to trace [{DEFAULT_TRACE_EXPANSIONS}]: ").strip()
    limit = int(answer) if answer.isdigit() else DEFAULT_TRACE_EXPANSIONS
    print_result(problem, algorithms[name](problem, trace=True, max_expansions=limit))


def _visualize_if_available(problem: Problem, results: dict[str, Result]) -> None:
    """Offer results to a teammate's ev/viz.py once that module exists."""
    viz = _optional_module("ev.viz")
    if viz is not None and hasattr(viz, "visualize"):
        viz.visualize(problem, results)


def menu() -> None:
    """Run the interactive menu until the user quits or input ends."""
    algorithms = available_algorithms()
    actions = ["show map", "run one algorithm with trace", "compare all algorithms",
               "run experiments", "final verdict", "quit"]
    try:
        while True:
            print("\nEV TRIP PLANNER")
            action = _ask_choice("Choose", actions)
            if action == "quit":
                return
            if action == "show map":
                show_map(build_problem(_ask_choice("Map", list(MAP_CHOICES))))
            elif action == "run one algorithm with trace":
                _run_with_trace(algorithms)
            elif action == "compare all algorithms":
                problem = build_problem(_ask_choice("Map", list(MAP_CHOICES)))
                results = compare(problem, algorithms)
                print_comparison(problem, results)
                _visualize_if_available(problem, results)
            elif action == "run experiments":
                run_experiments()
            else:
                verdict(algorithms)
    except EOFError:
        print()


def demo() -> None:
    """Show each feature without input, for a quick live demonstration."""
    algorithms = available_algorithms()
    print(f"Algorithms available: {', '.join(algorithms)}")
    show_map(toy_problem())
    print("\nBFS on the toy map, with trace:")
    print_result(toy_problem(), algorithms["BFS"](toy_problem(), trace=True))
    show_map(corridor_problem())
    results = compare(corridor_problem(), algorithms)
    print_comparison(corridor_problem(), results)
    verdict(algorithms)


def main() -> None:
    """Choose the mode from the command-line arguments."""
    if "--demo" in sys.argv[1:]:
        demo()
    elif "--experiments" in sys.argv[1:]:
        run_experiments()
    else:
        menu()


if __name__ == "__main__":
    main()
