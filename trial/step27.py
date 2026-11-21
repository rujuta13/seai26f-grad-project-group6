"""Step 27: choose. Pick the customers to target when test-mailing at most `budget`
people, and optionally optimise one goal (for example NUMSTOREPURCHASES+ for store
visits instead of NUMWEBPURCHASES+ for web orders).

The search is step17.choose: farthest-point exploration over the encoded inputs,
then an adaptive search that exploits and explores. This wrapper also enforces the
budget on label(), so it is never called more than `budget` times.
"""
from step17 import choose as _core_choose


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found while calling label() at most `budget` times.

    goal: optional goal column name (e.g. 'NUMSTOREPURCHASES+'). If given, only that
    goal is optimised. Otherwise all goals are combined as in step1.
    """
    budget = int(budget)
    if budget < 1:
        raise ValueError("budget must be at least 1")

    calls = 0

    def counted(i):
        nonlocal calls
        if calls >= budget:
            raise RuntimeError(f"label() may be called at most {budget} times")
        calls += 1
        return label(i)

    return _core_choose(blind_csv, counted, budget=budget, goal=goal)

