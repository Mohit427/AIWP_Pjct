"""Experiment runner for statistical evaluation of local search algorithms.

Runs multiple seeded experiments, collects statistics, and produces comparison tables.
"""

from __future__ import annotations

import inspect
import json
import random
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Callable

import pandas as pd

from ev.charger_placement import (
    ChargerPlacementProblem,
    Trip,
    generate_sample_trips,
    brute_force_optimum,
    BruteForceResult,
)
from ev.local_search import (
    steepest_ascent_hill_climbing,
    first_choice_hill_climbing,
    random_restart_hill_climbing,
    simulated_annealing,
    random_walk,
    LOCAL_SEARCH_ALGORITHMS,
)
from ev.maps import RoadMap, generate_sized_map, toy_map, corridor_map
from ev.model import MAX_BATTERY_CAP


# Fixed seeds for reproducibility across all experiments
DEFAULT_SEEDS = list(range(1001, 1031))  # 30 seeds


@dataclass
class AlgorithmRunResult:
    """Result of a single algorithm run with a specific seed."""
    algorithm: str
    seed: int
    best_value: float
    best_state: frozenset[str]
    iterations: int
    runtime_s: float
    extra: dict = None  # For algorithm-specific data


@dataclass
class AlgorithmStats:
    """Aggregated statistics for an algorithm over multiple seeds."""
    algorithm: str
    num_runs: int
    best: float
    worst: float
    mean: float
    median: float
    std_dev: float
    success_rate: float  # % runs reaching global optimum (if known)
    mean_runtime: float
    mean_iterations: float


@dataclass
class ExperimentResult:
    """Complete results for one experiment configuration."""
    map_name: str
    map_size: str
    num_candidates: int
    num_chargers: int
    num_trips: int
    brute_force: BruteForceResult | None
    algorithm_stats: list[AlgorithmStats]
    raw_runs: list[AlgorithmRunResult]


def create_problem_for_map(
    road_map: RoadMap,
    map_name: str,
    num_chargers: int,
    num_trips: int,
    candidate_fraction: float = 0.5,
    seed: int = 42,
) -> ChargerPlacementProblem:
    """Create a ChargerPlacementProblem for a given map.

    Candidates are a subset of nodes (non-charger nodes by default).
    """
    rng = random.Random(seed)

    # Get all nodes that could be candidates (non-charger nodes for fairness)
    all_nodes = list(road_map.coords.keys())
    existing_chargers = set(road_map.chargers.keys())

    # Candidates: nodes without chargers (so we're adding new ones)
    potential_candidates = [n for n in all_nodes if n not in existing_chargers]

    # If not enough, include some existing charger nodes too
    if len(potential_candidates) < num_chargers:
        potential_candidates = all_nodes

    # Select candidate subset
    num_candidates = max(num_chargers * 2, int(len(potential_candidates) * candidate_fraction))
    num_candidates = min(num_candidates, len(potential_candidates))
    candidates = rng.sample(potential_candidates, num_candidates)

    # Generate sample trips
    sample_trips = generate_sample_trips(road_map, num_trips, seed=seed + 1000)

    return ChargerPlacementProblem(
        road_map=road_map,
        candidates=candidates,
        num_chargers=num_chargers,
        sample_trips=sample_trips,
    )


def run_algorithm_once(
    algorithm_name: str,
    algorithm_fn: Callable,
    problem: ChargerPlacementProblem,
    seed: int,
    **kwargs
) -> AlgorithmRunResult:
    """Run a single algorithm with a specific seed."""
    start_time = time.perf_counter()

    # Merge default kwargs with provided ones
    default_kwargs = {
        "trace": False,
        "seed": seed,
    }
    default_kwargs.update(kwargs)

    # Filter kwargs to only include parameters accepted by the function
    sig = inspect.signature(algorithm_fn)
    filtered_kwargs = {k: v for k, v in default_kwargs.items() if k in sig.parameters}

    result = algorithm_fn(problem, **filtered_kwargs)

    runtime = time.perf_counter() - start_time

    return AlgorithmRunResult(
        algorithm=algorithm_name,
        seed=seed,
        best_value=result.best_value,
        best_state=result.best_state,
        iterations=result.iterations,
        runtime_s=runtime,
        extra={
            "stopped_reason": getattr(result, 'stopped_reason', None),
            "final_temp": getattr(result, 'final_temp', None),
            "acceptance_rate": getattr(result, 'acceptance_rate', None),
            "cooling_schedule": getattr(result, 'cooling_schedule', None),
            "num_restarts": getattr(result, 'num_restarts', None),
        }
    )


def compute_stats(
    algorithm: str,
    runs: list[AlgorithmRunResult],
    optimal_value: float | None = None,
    tolerance: float = 1e-6,
) -> AlgorithmStats:
    """Compute aggregated statistics from multiple runs."""
    values = [r.best_value for r in runs]
    runtimes = [r.runtime_s for r in runs]
    iterations = [r.iterations for r in runs]

    success_rate = 0.0
    if optimal_value is not None:
        successes = sum(1 for v in values if abs(v - optimal_value) < tolerance)
        success_rate = (successes / len(values)) * 100

    return AlgorithmStats(
        algorithm=algorithm,
        num_runs=len(runs),
        best=min(values),
        worst=max(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
        success_rate=success_rate,
        mean_runtime=statistics.mean(runtimes),
        mean_iterations=statistics.mean(iterations),
    )


def run_experiment(
    map_name: str,
    road_map: RoadMap,
    map_size: str,
    num_chargers: int,
    num_trips: int = 10,
    seeds: list[int] | None = None,
    algorithms: dict[str, Callable] | None = None,
    candidate_fraction: float = 0.5,
    brute_force: bool = False,
    **algorithm_kwargs
) -> ExperimentResult:
    """Run a complete experiment on one map with multiple algorithms and seeds."""
    if seeds is None:
        seeds = DEFAULT_SEEDS

    if algorithms is None:
        algorithms = {
            "Steepest-Ascent HC": steepest_ascent_hill_climbing,
            "First-Choice HC": first_choice_hill_climbing,
            "Random-Restart Steepest (R=20)": lambda problem, **kw: random_restart_hill_climbing(problem, **kw, use_first_choice=False, num_restarts=20),
            "Random-Restart First-Choice (R=20)": lambda problem, **kw: random_restart_hill_climbing(problem, **kw, use_first_choice=True, num_restarts=20),
            "Simulated Annealing (Exp)": lambda problem, **kw: simulated_annealing(problem, **kw, cooling_schedule="exponential"),
            "Simulated Annealing (Log)": lambda problem, **kw: simulated_annealing(problem, **kw, cooling_schedule="logarithmic"),
            "Random Walk": random_walk,
        }

    print(f"\n{'='*60}")
    print(f"Experiment: {map_name} ({map_size})")
    print(f"Chargers to place: {num_chargers}, Sample trips: {num_trips}")
    print(f"Seeds: {len(seeds)} runs per algorithm")
    print(f"{'='*60}")

    # Create problem (same for all seeds)
    problem = create_problem_for_map(
        road_map, map_name, num_chargers, num_trips,
        candidate_fraction=candidate_fraction, seed=42
    )

    print(f"Candidates: {len(problem.candidates)} nodes")
    print(f"Sample trips: {[(t.start, t.goal) for t in problem.sample_trips]}")

    # Brute force for ground truth (if feasible)
    bf_result = None
    if brute_force and len(problem.candidates) <= 15 and num_chargers <= 5:
        print("\nRunning brute-force optimum...")
        bf_result = brute_force_optimum(problem)
        print(f"Global optimum: {bf_result.optimal_value:.2f} min")
        print(f"Optimal chargers: {sorted(bf_result.optimal_state)}")
        print(f"Brute-force time: {bf_result.runtime_s:.2f}s")
    else:
        print("\nSkipping brute-force (too many combinations)")

    # Run each algorithm for each seed
    all_runs: list[AlgorithmRunResult] = []
    algorithm_stats: list[AlgorithmStats] = []

    for algo_name, algo_fn in algorithms.items():
        print(f"\n  Running {algo_name}...")
        algo_runs = []

        for seed in seeds:
            run_result = run_algorithm_once(algo_name, algo_fn, problem, seed, **algorithm_kwargs)
            algo_runs.append(run_result)
            all_runs.append(run_result)

        # Compute statistics
        optimal_val = bf_result.optimal_value if bf_result else None
        stats = compute_stats(algo_name, algo_runs, optimal_val)
        algorithm_stats.append(stats)

        print(f"    Best: {stats.best:.2f}, Mean: {stats.mean:.2f} ± {stats.std_dev:.2f}, "
              f"Success: {stats.success_rate:.1f}%, Time: {stats.mean_runtime:.2f}s")

    result = ExperimentResult(
        map_name=map_name,
        map_size=map_size,
        num_candidates=len(problem.candidates),
        num_chargers=num_chargers,
        num_trips=num_trips,
        brute_force=bf_result,
        algorithm_stats=algorithm_stats,
        raw_runs=all_runs,
    )

    return result


def print_comparison_table(result: ExperimentResult) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'='*100}")
    print(f"RESULTS: {result.map_name} ({result.map_size}) - "
          f"N={result.num_chargers} chargers, M={result.num_candidates} candidates, "
          f"{result.num_trips} trips")
    print(f"{'='*100}")

    if result.brute_force:
        print(f"\nGlobal Optimum (Brute-Force): {result.brute_force.optimal_value:.2f} min")
        print(f"Optimal Set: {sorted(result.brute_force.optimal_state)}")

    header = f"{'Algorithm':<35} {'Best':>10} {'Mean':>10} {'StdDev':>10} {'Median':>10} {'Success%':>10} {'Time(s)':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for stats in result.algorithm_stats:
        print(f"{stats.algorithm:<35} {stats.best:>10.2f} {stats.mean:>10.2f} "
              f"{stats.std_dev:>10.2f} {stats.median:>10.2f} {stats.success_rate:>9.1f}% {stats.mean_runtime:>10.2f}")

    print()


def save_results(result: ExperimentResult, filename: str) -> None:
    """Save experiment results to JSON file."""
    # Convert to serializable format
    data = {
        "map_name": result.map_name,
        "map_size": result.map_size,
        "num_candidates": result.num_candidates,
        "num_chargers": result.num_chargers,
        "num_trips": result.num_trips,
        "brute_force": None,
        "algorithm_stats": [],
        "raw_runs": [],
    }

    if result.brute_force:
        data["brute_force"] = {
            "optimal_value": result.brute_force.optimal_value,
            "optimal_state": sorted(result.brute_force.optimal_state),
            "runtime_s": result.brute_force.runtime_s,
        }

    for stats in result.algorithm_stats:
        data["algorithm_stats"].append({
            "algorithm": stats.algorithm,
            "num_runs": stats.num_runs,
            "best": stats.best,
            "worst": stats.worst,
            "mean": stats.mean,
            "median": stats.median,
            "std_dev": stats.std_dev,
            "success_rate": stats.success_rate,
            "mean_runtime": stats.mean_runtime,
            "mean_iterations": stats.mean_iterations,
        })

    for run in result.raw_runs:
        data["raw_runs"].append({
            "algorithm": run.algorithm,
            "seed": run.seed,
            "best_value": run.best_value,
            "best_state": sorted(run.best_state),
            "iterations": run.iterations,
            "runtime_s": run.runtime_s,
            "extra": run.extra,
        })

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {filename}")


def run_all_experiments(
    seeds: list[int] | None = None,
    output_dir: str = "results",
) -> list[ExperimentResult]:
    """Run experiments on all map sizes."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    if seeds is None:
        seeds = DEFAULT_SEEDS

    results = []

    # Toy map (small, for debugging and brute-force)
    print("\n" + "="*60)
    print("TOY MAP EXPERIMENT")
    print("="*60)
    toy_road_map = toy_map()
    result = run_experiment(
        "toy",
        toy_road_map,
        "toy",
        num_chargers=2,
        num_trips=5,
        seeds=seeds[:10],  # Fewer seeds for toy
        brute_force=True,
        max_iterations=500,
        max_neighbors_per_iter=30,
        num_restarts=10,
        initial_temp=50.0,
    )
    print_comparison_table(result)
    save_results(result, os.path.join(output_dir, "toy_results.json"))
    results.append(result)

    # Corridor map (Chennai-Bengaluru style)
    print("\n" + "="*60)
    print("CORRIDOR MAP EXPERIMENT")
    print("="*60)
    corridor_road_map = corridor_map()
    result = run_experiment(
        "corridor",
        corridor_road_map,
        "corridor",
        num_chargers=5,
        num_trips=10,
        seeds=seeds,
        brute_force=False,  # Too many combinations
        max_iterations=1000,
        max_neighbors_per_iter=50,
        num_restarts=20,
        initial_temp=100.0,
    )
    print_comparison_table(result)
    save_results(result, os.path.join(output_dir, "corridor_results.json"))
    results.append(result)

    # Generated maps: small, medium, large
    for size in ["small", "medium", "large"]:
        print(f"\n{'='*60}")
        print(f"GENERATED {size.upper()} MAP EXPERIMENT")
        print(f"{'='*60}")
        gen_map = generate_sized_map(size)
        num_chargers = {"small": 3, "medium": 5, "large": 8}[size]
        num_trips = {"small": 8, "medium": 12, "large": 15}[size]

        result = run_experiment(
            f"random-{size}",
            gen_map.road_map,
            size,
            num_chargers=num_chargers,
            num_trips=num_trips,
            seeds=seeds,
            brute_force=(size == "small"),
            max_iterations=2000,
            max_neighbors_per_iter=50,
            num_restarts=20,
            initial_temp=100.0,
        )
        print_comparison_table(result)
        save_results(result, os.path.join(output_dir, f"{size}_results.json"))
        results.append(result)

    return results


def load_results(filename: str) -> dict:
    """Load experiment results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def results_to_dataframe(results: list[ExperimentResult]) -> pd.DataFrame:
    """Convert all experiment results to a single pandas DataFrame for analysis."""
    rows = []
    for exp in results:
        for stats in exp.algorithm_stats:
            rows.append({
                "map": exp.map_name,
                "map_size": exp.map_size,
                "num_chargers": exp.num_chargers,
                "num_candidates": exp.num_candidates,
                "num_trips": exp.num_trips,
                "algorithm": stats.algorithm,
                "best": stats.best,
                "mean": stats.mean,
                "std_dev": stats.std_dev,
                "median": stats.median,
                "success_rate": stats.success_rate,
                "mean_runtime": stats.mean_runtime,
                "mean_iterations": stats.mean_iterations,
                "global_optimum": exp.brute_force.optimal_value if exp.brute_force else None,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Run all experiments when executed directly
    print("Running all experiments...")
    all_results = run_all_experiments()
    print("\n\nAll experiments completed!")

    # Combined summary
    df = results_to_dataframe(all_results)
    print("\nCombined Summary:")
    print(df.to_string(index=False))

    # Save combined CSV
    df.to_csv("results/all_experiments.csv", index=False)
    print("\nCombined CSV saved to results/all_experiments.csv")