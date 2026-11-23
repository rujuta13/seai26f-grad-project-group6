"""step17.py: choose a build to measure within a fixed scouting budget.

Both client questions use the same search, so choose() delegates to step7.choose,
which spends the budget on spread-out sampling and then on refining near the best
build found so far.

Question 1 (best strong penalty-taker, 20 scouts): call with budget=20 and goal=None,
so Penalties+ and Strength+ are both optimised and the answer is on their joint
Pareto front.

Question 2 (best under 24 in a league already scouted): the blind file should contain
only rows whose Age is under 24 and whose League is one the club already scouts.
Filtering candidates before measuring costs no budget, so label() is never called
for players outside that pool.

This wrapper adds a second, independent guard on the number of label() calls, so the
budget is enforced here as well as in step7.
"""
from __future__ import annotations

from step7 import choose as _choose


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found, after calling label(i) at most `budget` times.

    blind_csv: MOOT csv whose goal cells are all '?'.
    label(i): returns the goals of row i as a dict {goal column name: value}.
    budget: maximum number of label() calls; a positive integer.
    goal: optional name of one goal column; if given, only that goal is optimised.
    """
    if isinstance(budget, bool) or not float(budget).is_integer() or budget < 1:
        raise ValueError("budget must be a positive integer")
    budget = int(budget)

    calls = [0]

    def counted(i):
        if calls[0] >= budget:
            raise RuntimeError("measurement budget exceeded")
        calls[0] += 1
        return label(i)

    return int(_choose(blind_csv, counted, budget=budget, goal=goal))

