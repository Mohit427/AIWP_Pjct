"""Compatibility alias for `ev/viz.py`."""

from __future__ import annotations

from ev.viz import (
    ReconstructedRoute,
    RouteStep,
    format_placement_terminal,
    format_playback_terminal,
    format_route_terminal,
    generate_all_visuals,
    reconstruct_route,
    render_memory_comparison,
    render_placement_comparison,
    render_route_map,
    render_search_playback,
    visualize,
)

__all__ = [
    "RouteStep",
    "ReconstructedRoute",
    "reconstruct_route",
    "format_route_terminal",
    "format_playback_terminal",
    "format_placement_terminal",
    "render_route_map",
    "render_search_playback",
    "render_placement_comparison",
    "render_memory_comparison",
    "visualize",
    "generate_all_visuals",
]