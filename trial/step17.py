from step7 import choose as _choose


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Find a good row by benchmarking at most `budget` builds.

    blind_csv has every goal cell set to '?'. label(i) returns the goals of
    row i as a dict and is called at most `budget` times; a further call
    raises RuntimeError. If `goal` names one goal, only that goal is
    optimised. Returns the index of the best row found.
    """
    budget = int(budget)
    if budget < 1:
        raise ValueError("budget must be at least 1")

    calls = 0

    def guarded_label(i):
        nonlocal calls
        if calls >= budget:
            raise RuntimeError("benchmark budget exceeded")
        calls += 1
        return label(i)

    return int(_choose(blind_csv, guarded_label, budget=budget, goal=goal))

