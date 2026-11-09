"""Visualization suite for EV Trip Planner (Matplotlib + Terminal views).

Author: Rohiit
Deliverables:
  - Screen 1 (Mon 9 Nov): Map and route renderer reading the standard Result
    record (`render_route_map`, `format_route_terminal`).
  - Screen 2 (Wed 11 Nov): Step-by-step search playback showing explored nodes
    in expansion order from `Result.expanded_order` (`render_search_playback`,
    `format_playback_terminal`).
  - Screen 3 (Wed 11 Nov): Charger placement before-and-after view comparing
    initial vs optimized charger sets and convergence (`render_placement_comparison`,
    `format_placement_terminal`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # Safe non-interactive backend for headless/CLI and file export
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from ev.charger_placement import (
    ChargerPlacementProblem,
    LocalSearchResult,
    evaluate_charger_set,
)
from ev.metrics import SOLVED, Result
from ev.model import Action, CHARGE, DRIVE, Problem, RoadMap, State


@dataclass(frozen=True)
class RouteStep:
    """One reconstructed step along a solved route for plotting and terminal display."""

    step_index: int
    action: Action | None  # None for the initial state at step 0
    node: str
    battery_after: int
    step_minutes: int
    cumulative_minutes: int


@dataclass(frozen=True)
class ReconstructedRoute:
    """Full trajectory reconstructed by replaying a Result's path on a Problem."""

    steps: list[RouteStep]
    driven_edges: list[tuple[str, str]]
    charge_stops: dict[str, tuple[int, int]]  # node -> (units_charged, minutes_charged)
    visited_nodes: list[str]


def reconstruct_route(problem: Problem, result: Result) -> ReconstructedRoute:
    """Replay `result.path` against `problem` to extract node, battery, and charge stops."""
    start_node, start_bat = problem.initial_state()
    steps: list[RouteStep] = [
        RouteStep(
            step_index=0,
            action=None,
            node=start_node,
            battery_after=start_bat,
            step_minutes=0,
            cumulative_minutes=0,
        )
    ]
    driven_edges: list[tuple[str, str]] = []
    charge_stops: dict[str, tuple[int, int]] = {}
    visited_nodes: list[str] = [start_node]

    if not result.found or not result.path:
        return ReconstructedRoute(steps, driven_edges, charge_stops, visited_nodes)

    state: State = (start_node, start_bat)
    cum_min = 0

    for idx, action in enumerate(result.path, start=1):
        matched = False
        for candidate, next_state, step_min in problem.successors(state):
            if candidate == action:
                cum_min += step_min
                if action.kind == DRIVE:
                    driven_edges.append((state[0], next_state[0]))
                    visited_nodes.append(next_state[0])
                elif action.kind == CHARGE:
                    prev_units, prev_min = charge_stops.get(state[0], (0, 0))
                    charge_stops[state[0]] = (prev_units + 1, prev_min + step_min)
                state = next_state
                steps.append(
                    RouteStep(
                        step_index=idx,
                        action=action,
                        node=state[0],
                        battery_after=state[1],
                        step_minutes=step_min,
                        cumulative_minutes=cum_min,
                    )
                )
                matched = True
                break
        if not matched:
            raise ValueError(f"Step {idx}: illegal action {action} from state {state}")

    return ReconstructedRoute(steps, driven_edges, charge_stops, visited_nodes)


# ---------------------------------------------------------------------------
# Clean Terminal Renderers (Interface Option A / CLI Summary)
# ---------------------------------------------------------------------------


def format_route_terminal(problem: Problem, result: Result) -> str:
    """Return a clean, human-readable terminal view of the map route and Result metrics."""
    rm = problem.road_map
    lines = [
        "=" * 76,
        f"ROUTE MAP & PLAN SUMMARY | Algorithm: {result.algorithm} | Map: {rm.name}",
        "=" * 76,
        f"Trip: {problem.start} -> {problem.destination} | "
        f"Start Battery: {problem.start_battery}/{problem.battery_cap} units | "
        f"Status: {result.status.upper()}",
        f"Metrics: Cost = {result.total_minutes} min | Actions = {result.num_actions} | "
        f"Expanded = {result.nodes_expanded} | Peak Frontier = {result.peak_frontier} | "
        f"Time = {result.runtime_s * 1000:.2f} ms"
        + (
            f" | Peak Mem = {result.peak_memory_kb:.1f} KB"
            if result.peak_memory_kb is not None
            else ""
        ),
        "-" * 76,
    ]

    if not result.found:
        lines.append("No route to display (status: " + result.status + ").")
        lines.append("=" * 76)
        return "\n".join(lines)

    recon = reconstruct_route(problem, result)
    lines.append(
        f"{'Step':>4}  {'Action':<24} {'Location':<18} {'Battery':>9} {'Step(m)':>8} {'Total(m)':>8}"
    )
    lines.append("-" * 76)
    for st in recon.steps:
        act_str = "START" if st.action is None else str(st.action)
        bat_bar = f"{st.battery_after:>2}/{problem.battery_cap:<2}"
        lines.append(
            f"{st.step_index:>4}  {act_str:<24} {st.node:<18} {bat_bar:>9} "
            f"{st.step_minutes:>8} {st.cumulative_minutes:>8}"
        )

    if recon.charge_stops:
        stops_desc = ", ".join(
            f"{node} (+{u}u, {m}m)" for node, (u, m) in recon.charge_stops.items()
        )
        lines.append("-" * 76)
        lines.append(f"Charging Stops: {stops_desc}")

    lines.append("=" * 76)
    return "\n".join(lines)


def format_playback_terminal(
    problem: Problem, result: Result, max_steps: int = 30
) -> str:
    """Return a step-by-step terminal playback of `result.expanded_order`."""
    lines = [
        "=" * 76,
        f"SEARCH PLAYBACK | Algorithm: {result.algorithm} | "
        f"Total Expanded: {result.nodes_expanded} (Logged: {len(result.expanded_order)})",
        "=" * 76,
        f"{'Order':>6}  {'Node':<20} {'Battery':>8} {'Dist to Goal (km)':>18} {'Charger?':>10}",
        "-" * 76,
    ]
    rm = problem.road_map
    shown = result.expanded_order[:max_steps]
    for idx, (node, bat) in enumerate(shown, start=1):
        dist = rm.straight_line_km(node, problem.destination)
        has_chg = "YES" if node in rm.chargers else "-"
        lines.append(
            f"{idx:>6}  {node:<20} {bat:>8} {dist:>18.1f} {has_chg:>10}"
        )
    if len(result.expanded_order) > max_steps:
        lines.append(
            f"... ({len(result.expanded_order) - max_steps} additional expansions omitted) ..."
        )
    lines.append("=" * 76)
    return "\n".join(lines)


def format_placement_terminal(
    problem: ChargerPlacementProblem,
    initial_state: frozenset[str],
    result: LocalSearchResult,
) -> str:
    """Return a clean terminal summary of Charger Placement Before vs After."""
    before_val = evaluate_charger_set(problem, initial_state)
    after_val = result.best_value
    improvement = before_val - after_val
    pct = (improvement / before_val * 100.0) if before_val > 0 else 0.0
    lines = [
        "=" * 76,
        f"CHARGER PLACEMENT BEFORE & AFTER | Algorithm: {result.algorithm}",
        "=" * 76,
        f"Map: {problem.road_map.name} | Candidates (M): {len(problem.candidates)} | "
        f"Chargers to Place (N): {problem.num_chargers} | Sample Trips: {len(problem.sample_trips)}",
        f"BEFORE (Initial Sites): {sorted(initial_state)} -> Avg Trip Time: {before_val:.2f} min",
        f"AFTER  (Best Sites):    {sorted(result.best_state)} -> Avg Trip Time: {after_val:.2f} min",
        f"Improvement: -{improvement:.2f} min ({pct:.1f}% faster) | "
        f"Iterations: {result.iterations} | Runtime: {result.runtime_s * 1000:.2f} ms",
        "=" * 76,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matplotlib Screen 1: Route Map & Battery Profile Renderer
# ---------------------------------------------------------------------------


def _draw_base_road_network(
    ax: plt.Axes,
    road_map: RoadMap,
    annotate_edges: bool = True,
    annotate_nodes: bool = True,
    alpha_edges: float = 0.45,
) -> None:
    """Draw all road edges and background nodes onto `ax`."""
    drawn_pairs: set[tuple[str, str]] = set()
    for a, links in road_map.roads.items():
        ax_km, ay_km = road_map.coords[a]
        for b, road in links.items():
            pair = (min(a, b), max(a, b))
            if pair in drawn_pairs:
                continue
            drawn_pairs.add(pair)
            bx_km, by_km = road_map.coords[b]
            ax.plot(
                [ax_km, bx_km],
                [ay_km, by_km],
                color="#94a3b8",
                linewidth=1.3,
                alpha=alpha_edges,
                zorder=1,
            )
            if annotate_edges and road_map.num_nodes() <= 25:
                mx, my = (ax_km + bx_km) / 2.0, (ay_km + by_km) / 2.0
                ax.text(
                    mx,
                    my,
                    f"{road.minutes}m\n({road.battery_cost}u)",
                    fontsize=6.5,
                    color="#475569",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.12",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.75,
                    ),
                    zorder=2,
                )

    if annotate_nodes and road_map.num_nodes() <= 35:
        for node, (nx, ny) in road_map.coords.items():
            ax.annotate(
                node,
                (nx, ny),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                fontweight="bold" if road_map.num_nodes() <= 25 else "normal",
                color="#1e293b",
                zorder=6,
            )


def render_route_map(
    problem: Problem,
    result: Result,
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Render Screen 1: Map & Route view with active charging stops and battery profile.

    Reads the standard `Result` record (`path`, `total_minutes`, `nodes_expanded`,
    `peak_frontier`, `runtime_s`) and `Problem` to produce a two-panel figure:
      - Left/Top panel: Spatial road network, charger stations, start/destination,
        highlighted route edges with directional arrows, and charging stop callouts.
      - Right/Bottom panel: Step-by-step battery state-of-charge profile along the route.
    """
    rm = problem.road_map
    recon = reconstruct_route(problem, result)

    fig, (ax_map, ax_bat) = plt.subplots(
        1,
        2,
        figsize=(15, 6.8),
        gridspec_kw={"width_ratios": [1.65, 1.0]},
        dpi=150,
    )
    fig.patch.set_facecolor("#f8fafc")
    for ax in (ax_map, ax_bat):
        ax.set_facecolor("#ffffff")

    _draw_base_road_network(
        ax_map,
        rm,
        annotate_edges=(rm.num_nodes() <= 25),
        annotate_nodes=(rm.num_nodes() <= 35),
    )

    # Plot regular nodes vs charger nodes
    regular_nodes = [
        n
        for n in rm.coords
        if n not in rm.chargers and n not in (problem.start, problem.destination)
    ]
    charger_nodes = [n for n in rm.chargers]

    if regular_nodes:
        rx = [rm.coords[n][0] for n in regular_nodes]
        ry = [rm.coords[n][1] for n in regular_nodes]
        ax_map.scatter(
            rx,
            ry,
            s=65,
            c="#cbd5e1",
            edgecolors="#475569",
            linewidths=1.2,
            label="City / Junction",
            zorder=3,
        )

    if charger_nodes:
        cx = [rm.coords[n][0] for n in charger_nodes]
        cy = [rm.coords[n][1] for n in charger_nodes]
        ax_map.scatter(
            cx,
            cy,
            s=115,
            marker="D",
            c="#38bdf8",
            edgecolors="#0369a1",
            linewidths=1.5,
            label="Charging Station",
            zorder=4,
        )

    # Highlight Start and Destination
    sx, sy = rm.coords[problem.start]
    gx, gy = rm.coords[problem.destination]
    ax_map.scatter(
        [sx],
        [sy],
        s=190,
        marker="^",
        c="#22c55e",
        edgecolors="#14532d",
        linewidths=1.8,
        label=f"Start ({problem.start})",
        zorder=5,
    )
    ax_map.scatter(
        [gx],
        [gy],
        s=220,
        marker="*",
        c="#ef4444",
        edgecolors="#7f1d1d",
        linewidths=1.6,
        label=f"Goal ({problem.destination})",
        zorder=5,
    )

    # Highlight route edges and active charging stops if solved
    if result.found:
        for u, v in recon.driven_edges:
            ux, uy = rm.coords[u]
            vx, vy = rm.coords[v]
            ax_map.plot(
                [ux, vx],
                [uy, vy],
                color="#2563eb",
                linewidth=3.2,
                alpha=0.9,
                zorder=4,
            )
            arrow = FancyArrowPatch(
                (ux, uy),
                (vx, vy),
                arrowstyle="-|>",
                mutation_scale=13,
                color="#1d4ed8",
                linewidth=1.8,
                zorder=5,
            )
            ax_map.add_patch(arrow)

        if recon.charge_stops:
            used_x = [rm.coords[n][0] for n in recon.charge_stops]
            used_y = [rm.coords[n][1] for n in recon.charge_stops]
            ax_map.scatter(
                used_x,
                used_y,
                s=185,
                marker="D",
                c="#f59e0b",
                edgecolors="#78350f",
                linewidths=2.0,
                label="Active Charge Stop",
                zorder=6,
            )
            for node, (units, mins) in recon.charge_stops.items():
                nx, ny = rm.coords[node]
                ax_map.annotate(
                    f"+{units}u ({mins}m)",
                    (nx, ny),
                    textcoords="offset points",
                    xytext=(0, -15),
                    ha="center",
                    fontsize=7.5,
                    fontweight="bold",
                    color="#92400e",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="#fef3c7",
                        edgecolor="#f59e0b",
                        alpha=0.95,
                    ),
                    zorder=7,
                )

    status_str = (
        f"Cost: {result.total_minutes} min | Actions: {result.num_actions} | "
        f"Expanded: {result.nodes_expanded} | Peak Frontier: {result.peak_frontier}"
        if result.found
        else f"Status: {result.status.upper()} | Expanded: {result.nodes_expanded}"
    )
    ax_map.set_title(
        f"Route Map — {result.algorithm} on '{rm.name}'\n{status_str}",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
        pad=10,
    )
    ax_map.set_xlabel("East–West Coordinate (km)", fontsize=9.5)
    ax_map.set_ylabel("North–South Coordinate (km)", fontsize=9.5)
    ax_map.grid(True, linestyle="--", alpha=0.3)
    ax_map.legend(loc="best", fontsize=8, framealpha=0.92)

    # Right panel: Battery State-of-Charge profile along the route
    if result.found and recon.steps:
        xs = [st.cumulative_minutes for st in recon.steps]
        ys = [st.battery_after for st in recon.steps]
        ax_bat.plot(
            xs,
            ys,
            marker="o",
            color="#0284c7",
            linewidth=2.4,
            markersize=6,
            label="Battery Level (units)",
        )
        ax_bat.fill_between(xs, ys, color="#bae6fd", alpha=0.35)
        ax_bat.axhline(
            problem.battery_cap,
            color="#64748b",
            linestyle=":",
            linewidth=1.2,
            label=f"Battery Cap ({problem.battery_cap}u)",
        )
        ax_bat.axhline(0, color="#ef4444", linestyle="--", linewidth=1.0, alpha=0.7)

        for st in recon.steps:
            if st.action is None or st.action.kind == DRIVE:
                ax_bat.annotate(
                    st.node,
                    (st.cumulative_minutes, st.battery_after),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=7.5,
                    rotation=20 if len(recon.steps) > 8 else 0,
                    color="#1e293b",
                )
            elif st.action.kind == CHARGE:
                ax_bat.scatter(
                    [st.cumulative_minutes],
                    [st.battery_after],
                    c="#f59e0b",
                    s=70,
                    zorder=5,
                )

        ax_bat.set_ylim(-0.5, problem.battery_cap + 1.5)
        ax_bat.set_xlabel("Elapsed Trip Time (minutes)", fontsize=9.5)
        ax_bat.set_ylabel("Battery Charge (units)", fontsize=9.5)
        ax_bat.set_title(
            "Battery State-of-Charge vs. Elapsed Time",
            fontsize=11,
            fontweight="bold",
            color="#0f172a",
            pad=10,
        )
        ax_bat.grid(True, linestyle="--", alpha=0.35)
        ax_bat.legend(loc="upper right", fontsize=8)
    else:
        ax_bat.text(
            0.5,
            0.5,
            f"No valid route found\n({result.status})",
            ha="center",
            va="center",
            fontsize=12,
            color="#64748b",
            transform=ax_bat.transAxes,
        )
        ax_bat.set_title("Battery Profile (Unavailable)", fontsize=11, fontweight="bold")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Matplotlib Screen 2: Step-by-Step Search Playback
# ---------------------------------------------------------------------------


def render_search_playback(
    problem: Problem,
    result: Result,
    num_snapshots: int = 4,
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Render Screen 2: Step-by-step search playback from `result.expanded_order`.

    Displays progressive snapshots (e.g. 25%, 50%, 75%, 100% of expansions) so
    the viewer can see the exact order in which states are explored across the
    road network and how the frontier progresses toward the destination.
    """
    rm = problem.road_map
    expanded = result.expanded_order
    total_logged = len(expanded)

    if num_snapshots < 1:
        raise ValueError("num_snapshots must be >= 1")

    cols = min(2, num_snapshots)
    rows = (num_snapshots + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6.8 * cols, 5.2 * rows),
        dpi=150,
        squeeze=False,
    )
    fig.patch.set_facecolor("#f8fafc")
    flat_axes = [ax for row in axes for ax in row]

    cmap = plt.cm.plasma

    for idx in range(num_snapshots):
        ax = flat_axes[idx]
        ax.set_facecolor("#ffffff")
        frac = (idx + 1) / num_snapshots
        cutoff = max(1, round(total_logged * frac)) if total_logged > 0 else 0
        sub_expanded = expanded[:cutoff]

        _draw_base_road_network(
            ax,
            rm,
            annotate_edges=False,
            annotate_nodes=(rm.num_nodes() <= 25),
            alpha_edges=0.35,
        )

        # Unexplored nodes in light grey
        all_x = [rm.coords[n][0] for n in rm.coords]
        all_y = [rm.coords[n][1] for n in rm.coords]
        ax.scatter(
            all_x,
            all_y,
            s=55,
            c="#e2e8f0",
            edgecolors="#94a3b8",
            linewidths=1.0,
            zorder=2,
        )

        # First expansion index and expansion count per node in this snapshot
        first_seen: dict[str, int] = {}
        visit_counts: dict[str, int] = {}
        for step_i, (node, _bat) in enumerate(sub_expanded, start=1):
            if node not in first_seen:
                first_seen[node] = step_i
            visit_counts[node] = visit_counts.get(node, 0) + 1

        if first_seen:
            nodes_seen = list(first_seen.keys())
            xs = [rm.coords[n][0] for n in nodes_seen]
            ys = [rm.coords[n][1] for n in nodes_seen]
            orders = [first_seen[n] for n in nodes_seen]
            sizes = [80 + 22 * min(visit_counts[n], 8) for n in nodes_seen]
            sc = ax.scatter(
                xs,
                ys,
                s=sizes,
                c=orders,
                cmap=cmap,
                vmin=1,
                vmax=max(1, total_logged),
                edgecolors="#0f172a",
                linewidths=1.2,
                zorder=4,
            )
            if idx == num_snapshots - 1:
                cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("First Expansion Step #", fontsize=8.5)

        # Overlay final route on the 100% snapshot if solved
        if idx == num_snapshots - 1 and result.found:
            recon = reconstruct_route(problem, result)
            for u, v in recon.driven_edges:
                ux, uy = rm.coords[u]
                vx, vy = rm.coords[v]
                ax.plot(
                    [ux, vx],
                    [uy, vy],
                    color="#16a34a",
                    linewidth=2.8,
                    alpha=0.9,
                    zorder=5,
                )

        # Start and Goal markers
        sx, sy = rm.coords[problem.start]
        gx, gy = rm.coords[problem.destination]
        ax.scatter(
            [sx], [sy], s=150, marker="^", c="#22c55e", edgecolors="#14532d", zorder=6
        )
        ax.scatter(
            [gx], [gy], s=175, marker="*", c="#ef4444", edgecolors="#7f1d1d", zorder=6
        )

        last_state_str = (
            f"Last: {sub_expanded[-1][0]} (bat={sub_expanded[-1][1]})"
            if sub_expanded
            else "No expansions"
        )
        ax.set_title(
            f"Playback {int(frac * 100)}% — Step {cutoff}/{total_logged}\n"
            f"Unique Cities Reached: {len(first_seen)}/{rm.num_nodes()} | {last_state_str}",
            fontsize=9.5,
            fontweight="bold",
            color="#0f172a",
        )
        ax.set_xlabel("x (km)", fontsize=8.5)
        ax.set_ylabel("y (km)", fontsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.25)

    for extra_ax in flat_axes[num_snapshots:]:
        extra_ax.axis("off")

    fig.suptitle(
        f"Step-by-Step Search Playback — {result.algorithm} on '{rm.name}' "
        f"(Total Expanded: {result.nodes_expanded}, Peak Frontier: {result.peak_frontier})",
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
        y=1.01,
    )
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Matplotlib Screen 3: Charger Placement Before-and-After View
# ---------------------------------------------------------------------------


def render_placement_comparison(
    problem: ChargerPlacementProblem,
    initial_state: frozenset[str],
    result: LocalSearchResult,
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Render Screen 3: Charger placement Before (initial) vs After (optimized).

    Left panel: Initial charger sites (`initial_state`) and average sample trip time.
    Center panel: Optimized charger sites (`result.best_state`) and improved average time.
    Right panel: Objective value convergence curve across iterations from `trace_history`.
    """
    rm = problem.road_map
    before_val = evaluate_charger_set(problem, initial_state)
    after_val = result.best_value

    has_trace = bool(result.trace_history)
    ncols = 3 if has_trace else 2
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(6.2 * ncols, 5.8),
        dpi=150,
    )
    fig.patch.set_facecolor("#f8fafc")

    def draw_placement_panel(
        ax: plt.Axes,
        chosen_sites: frozenset[str],
        panel_title: str,
        highlight_color: str,
    ) -> None:
        ax.set_facecolor("#ffffff")
        _draw_base_road_network(
            ax,
            rm,
            annotate_edges=False,
            annotate_nodes=(rm.num_nodes() <= 25),
            alpha_edges=0.35,
        )

        # Draw sample trip desire lines faintly
        for trip in problem.sample_trips:
            tx1, ty1 = rm.coords[trip.start]
            tx2, ty2 = rm.coords[trip.goal]
            ax.plot(
                [tx1, tx2],
                [ty1, ty2],
                linestyle=":",
                color="#94a3b8",
                linewidth=0.9,
                alpha=0.4,
                zorder=1,
            )

        existing = set(rm.chargers.keys())
        candidates = set(problem.candidates)
        unchosen_candidates = candidates - set(chosen_sites)
        other_nodes = [
            n for n in rm.coords if n not in existing and n not in candidates
        ]

        if other_nodes:
            ax.scatter(
                [rm.coords[n][0] for n in other_nodes],
                [rm.coords[n][1] for n in other_nodes],
                s=50,
                c="#e2e8f0",
                edgecolors="#64748b",
                label="Regular Node",
                zorder=2,
            )
        if existing:
            ax.scatter(
                [rm.coords[n][0] for n in existing],
                [rm.coords[n][1] for n in existing],
                s=100,
                marker="s",
                c="#38bdf8",
                edgecolors="#0369a1",
                label="Existing Charger",
                zorder=3,
            )
        if unchosen_candidates:
            ax.scatter(
                [rm.coords[n][0] for n in unchosen_candidates],
                [rm.coords[n][1] for n in unchosen_candidates],
                s=90,
                marker="o",
                c="#fef08a",
                edgecolors="#ca8a04",
                linewidths=1.3,
                label="Unchosen Candidate",
                zorder=4,
            )
        if chosen_sites:
            ax.scatter(
                [rm.coords[n][0] for n in chosen_sites],
                [rm.coords[n][1] for n in chosen_sites],
                s=190,
                marker="D",
                c=highlight_color,
                edgecolors="#0f172a",
                linewidths=1.8,
                label=f"Selected Site (N={len(chosen_sites)})",
                zorder=5,
            )

        ax.set_title(panel_title, fontsize=10.5, fontweight="bold", color="#0f172a")
        ax.set_xlabel("x (km)", fontsize=8.5)
        ax.set_ylabel("y (km)", fontsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(loc="best", fontsize=7.5, framealpha=0.9)

    draw_placement_panel(
        axes[0],
        initial_state,
        f"BEFORE: Initial Placement\nAvg Trip Time = {before_val:.2f} min\nSites: {sorted(initial_state)}",
        "#f97316",
    )
    draw_placement_panel(
        axes[1],
        result.best_state,
        f"AFTER: {result.algorithm}\nAvg Trip Time = {after_val:.2f} min "
        f"(Δ = {after_val - before_val:+.2f} min)\nSites: {sorted(result.best_state)}",
        "#16a34a",
    )

    if has_trace:
        ax_conv = axes[2]
        ax_conv.set_facecolor("#ffffff")
        iters: list[int] = []
        curr_vals: list[float] = []
        running_best: list[float] = []
        best_so_far = float("inf")
        for idx, entry in enumerate(result.trace_history):
            if "current_value" in entry:
                it = int(entry.get("iteration", idx))
                val = float(entry["current_value"])
                best_so_far = min(best_so_far, val)
                iters.append(it)
                curr_vals.append(val)
                running_best.append(best_so_far)
            elif "best_value" in entry:
                it = int(entry.get("restart", idx))
                val = float(entry["best_value"])
                best_so_far = min(best_so_far, val)
                iters.append(it)
                curr_vals.append(val)
                running_best.append(best_so_far)

        if iters:
            ax_conv.plot(
                iters,
                curr_vals,
                color="#94a3b8",
                linewidth=1.3,
                alpha=0.75,
                label="Current State Cost",
            )
            ax_conv.plot(
                iters,
                running_best,
                color="#16a34a",
                linewidth=2.4,
                label="Best Cost So Far",
            )
        ax_conv.set_title(
            f"Optimization Trace ({result.iterations} iterations)",
            fontsize=10.5,
            fontweight="bold",
            color="#0f172a",
        )
        ax_conv.set_xlabel("Iteration / Step", fontsize=9)
        ax_conv.set_ylabel("Average Trip Time (minutes)", fontsize=9)
        ax_conv.grid(True, linestyle="--", alpha=0.35)
        ax_conv.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Charger Placement Optimization on '{rm.name}' — {result.algorithm}",
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
        y=1.02,
    )
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig