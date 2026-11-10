"""Memory-bounded informed search plug-in module for main.py (see CONTRACT.md).

Exposes `ALGORITHMS = {"IDA*": ida_star, "DFBB": dfbb}` so `main.py` discovers
and compares IDA* and DFBB automatically.
"""

from __future__ import annotations

from ev.memory_bounded import (
    ALGORITHMS,
    dfbb,
    dfbb_unbounded,
    ida_star,
    random_walk,
    run_memory_bounded_experiments,
    summarize_memory_bounded,
)

__all__ = [
    "ALGORITHMS",
    "ida_star",
    "dfbb",
    "dfbb_unbounded",
    "random_walk",
    "run_memory_bounded_experiments",
    "summarize_memory_bounded",
]
