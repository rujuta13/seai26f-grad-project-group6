"""Step 17: client-facing wrapper around step7.choose for budgeted scouting."""

from step7 import choose as _choose


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Pick the best row found by labelling at most `budget` rows.

    blind_csv has every goal cell set to '?'. label(i) returns the goals of
    row i as a dict and may be called at most `budget` times; a further call
    raises an error. Returns the index of the best row found. If `goal` names
    one goal, only that goal is optimised; otherwise the min-max normalised sum
    of all goals is used.

    Filters such as age or league are not part of this interface. To restrict
    the search, pass a blind_csv that already contains only the eligible rows
    (for example, players under 24 in leagues already scouted). Row indices
    returned refer to the rows of the file that was passed in.
    """
    if not isinstance(budget, int) or isinstance(budget, bool):
        raise TypeError("budget must be an integer")
    if budget < 0:
        raise ValueError("budget must be non-negative")

    calls = 0

    def counted_label(i):
        nonlocal calls
        if calls >= budget:
            raise RuntimeError("labelling budget exceeded")
        calls += 1
        return label(i)

    return _choose(blind_csv, counted_label, budget=budget, goal=goal)

