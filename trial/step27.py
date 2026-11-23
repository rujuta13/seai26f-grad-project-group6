from step17 import choose as _choose


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Find a good row by benchmarking at most `budget` builds.

    blind_csv has every goal cell set to '?'. label(i) returns the goals of
    row i as a dict and may be called at most `budget` times; a further call
    raises RuntimeError. If `goal` names one goal (for example
    'NUMSTOREPURCHASES+' for store visits, or 'NUMWEBPURCHASES+' for web
    orders), only that goal is optimised. Otherwise all goals are combined.
    Returns the index of the best row found.
    """
    return int(_choose(blind_csv, label, budget=budget, goal=goal))

