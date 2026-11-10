"""EV trip planner: shared model, maps, metrics, and search algorithms."""

from ev.model import (
    MAX_BATTERY_CAP,
    State,
    Action,
    Road,
    Charger,
    RoadMap,
    Problem,
    DRIVE,
    CHARGE,
)

from ev.maps import (
    KM_PER_BATTERY_UNIT,
    battery_cost_for_km,
    toy_map,
    toy_problem,
    corridor_map,
    corridor_problem,
    generate_map,
    generate_sized_map,
    seeded_start_goal_pairs,
    GeneratedMap,
    TOY_START,
    TOY_GOAL,
    CORRIDOR_START,
    CORRIDOR_GOAL,
)

from ev.metrics import (
    Result,
    SearchRun,
    measure,
    SOLVED,
    NO_SOLUTION,
    CUTOFF,
    DEFAULT_MAX_EXPANSIONS,
    DEFAULT_TIME_LIMIT_S,
)

from ev.uninformed import (
    bfs,
    dfs,
    iddfs,
    ALGORITHMS as UNINFORMED_ALGORITHMS,
)

from ev.heuristics import (
    GraphBounds,
    precompute_graph_bounds,
    h0_zero,
    h1_straight_line_time,
    h2_distance_plus_forced_charge,
    make_heuristics,
    make_problem_heuristics,
    HEURISTIC_NAMES,
)

from ev.informed import (
    ucs,
    greedy_best_first,
    astar,
    astar_h0,
    astar_h1,
    astar_h2,
    ALGORITHMS as INFORMED_ALGORITHMS,
)

from ev.memory_bounded import (
    ida_star,
    dfbb,
    dfbb_unbounded,
    random_walk,
    run_memory_bounded_experiments,
    summarize_memory_bounded,
    ALGORITHMS as MEMORY_BOUNDED_ALGORITHMS,
)

from ev.charger_placement import (
    Trip,
    ChargerPlacementProblem,
    evaluate_charger_set,
    generate_sample_trips,
    brute_force_optimum,
    BruteForceResult,
    LocalSearchResult,
    HillClimbingResult,
    RandomRestartResult,
    SimulatedAnnealingResult,
)

from ev.local_search import (
    steepest_ascent_hill_climbing,
    first_choice_hill_climbing,
    random_restart_hill_climbing,
    simulated_annealing,
    LOCAL_SEARCH_ALGORITHMS,
)

from ev.experiment_runner import (
    run_experiment,
    run_all_experiments,
    print_comparison_table,
    save_results,
    AlgorithmRunResult,
    AlgorithmStats,
    ExperimentResult,
    DEFAULT_SEEDS,
)

__all__ = [
    # Model
    "MAX_BATTERY_CAP",
    "State",
    "Action",
    "Road",
    "Charger",
    "RoadMap",
    "Problem",
    "DRIVE",
    "CHARGE",
    # Maps
    "KM_PER_BATTERY_UNIT",
    "battery_cost_for_km",
    "toy_map",
    "toy_problem",
    "corridor_map",
    "corridor_problem",
    "generate_map",
    "generate_sized_map",
    "seeded_start_goal_pairs",
    "GeneratedMap",
    "TOY_START",
    "TOY_GOAL",
    "CORRIDOR_START",
    "CORRIDOR_GOAL",
    # Metrics
    "Result",
    "SearchRun",
    "measure",
    "SOLVED",
    "NO_SOLUTION",
    "CUTOFF",
    "DEFAULT_MAX_EXPANSIONS",
    "DEFAULT_TIME_LIMIT_S",
    # Uninformed search
    "bfs",
    "dfs",
    "iddfs",
    "UNINFORMED_ALGORITHMS",
    # Heuristics & Informed search
    "GraphBounds",
    "precompute_graph_bounds",
    "h0_zero",
    "h1_straight_line_time",
    "h2_distance_plus_forced_charge",
    "make_heuristics",
    "make_problem_heuristics",
    "HEURISTIC_NAMES",
    "ucs",
    "greedy_best_first",
    "astar",
    "astar_h0",
    "astar_h1",
    "astar_h2",
    "INFORMED_ALGORITHMS",
    # Memory-bounded search
    "ida_star",
    "dfbb",
    "dfbb_unbounded",
    "run_memory_bounded_experiments",
    "summarize_memory_bounded",
    "MEMORY_BOUNDED_ALGORITHMS",
    # Charger placement
    "Trip",
    "ChargerPlacementProblem",
    "evaluate_charger_set",
    "generate_sample_trips",
    "brute_force_optimum",
    "BruteForceResult",
    "LocalSearchResult",
    "HillClimbingResult",
    "RandomRestartResult",
    "SimulatedAnnealingResult",
    # Local search
    "steepest_ascent_hill_climbing",
    "first_choice_hill_climbing",
    "random_restart_hill_climbing",
    "simulated_annealing",
    "random_walk",
    "LOCAL_SEARCH_ALGORITHMS",
    # Experiment runner
    "run_experiment",
    "run_all_experiments",
    "print_comparison_table",
    "save_results",
    "AlgorithmRunResult",
    "AlgorithmStats",
    "ExperimentResult",
    "DEFAULT_SEEDS",
]