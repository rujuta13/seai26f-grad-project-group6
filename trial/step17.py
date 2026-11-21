"""Step 17: scout-limited search for the best build, optionally on one goal.

choose(blind_csv, label, budget=30, goal=None) runs step7.choose and enforces
the scouting limit. label(i) may be called at most `budget` times. A call past
the limit raises RuntimeError instead of being answered.

For the client's requests, the caller sets the arguments:
  - "best strong penalty-taker, 20 players scouted": budget=20, goal="Penalties".
  - "best under 24 in a scouted league": blind_csv should contain only the
    under-24 players from leagues already scouted. The filter is applied
    before the file is passed in, so choose() does not need to know about
    age or league.

Returns the index of the best row found, as in step7.choose.
"""
from step7 import choose as _choose_step7


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found after at most `budget` calls to label().

    blind_csv: MOOT csv whose goal cells are all '?'.
    label(i): returns {goal column: value} for row i.
    budget: maximum number of label() calls. Exceeding it raises RuntimeError.
    goal: optional goal column (or its name without the +/- suffix) to optimise
          alone. If None, all goals are combined.

    Raises ValueError for budget < 1, an empty file, or an unknown goal.
    """
    if budget < 1:
        raise ValueError("budget must be at least 1")

    calls = 0

    def counted(i):
        nonlocal calls
        if calls >= budget:
            raise RuntimeError(f"label() called more than budget={budget} times")
        calls += 1
        return label(i)

    return _choose_step7(blind_csv, counted, budget=budget, goal=goal)

