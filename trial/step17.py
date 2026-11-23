"""step17.py: choose the best build while measuring at most `budget` builds.

Thin wrapper around step7.choose. It checks the arguments before any build is
measured, and it counts calls to `label` so that a call beyond `budget` raises
an error instead of silently measuring more builds than allowed.

The client's two questions map onto this function through its arguments: for
the penalty-taker question use goal="Penalties+" and budget=20. The age and
league restriction in the second question is a property of which rows are in
blind_csv, so the caller should prepare that file before calling choose.
"""

from step1 import _load
from step7 import choose as _choose


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found, calling label(i) at most `budget` times.

    blind_csv has every goal cell set to '?'. label(i) returns the goals of
    row i as a dict. If `goal` names one goal, only that goal is optimised.
    Raises RuntimeError if the search would call label more than `budget` times.
    """
    if budget < 1:
        raise ValueError("budget must be at least 1")

    _rows, _inputs, goals = _load(blind_csv)
    if not goals:
        raise ValueError("blind_csv has no goal columns")
    if goal is not None and goal not in {name for name, _ in goals}:
        raise ValueError(f"unknown goal {goal!r}")

    calls = [0]

    def counted_label(i):
        if calls[0] >= budget:
            raise RuntimeError(
                f"label called more than budget={budget} times"
            )
        calls[0] += 1
        return label(i)

    return int(_choose(blind_csv, counted_label, budget=budget, goal=goal))

