"""Step 27: choose the best customers to target within a limited test-mailing budget."""

from step17 import choose as _choose


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Pick the best row found by labelling at most `budget` rows.

    blind_csv has every goal cell set to '?'. label(i) returns the goals of
    row i as a dict and may be called at most `budget` times; a further call
    raises an error. Returns the index of the best row found. If `goal` names
    one goal, only that goal is optimised; otherwise the min-max normalised sum
    of all goals is used.
    """
    return _choose(blind_csv, label, budget=budget, goal=goal)

