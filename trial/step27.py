from step17 import choose as _choose


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Return the index of the best row found, labelling at most `budget` rows.

    blind_csv: CSV whose goal cells are all '?'.
    label(i):  returns a dict {goal column: value} for row i. Called at most
               `budget` times; a further call raises RuntimeError.
    goal:      optional name of one goal column to optimise. If None, all goal
               columns are optimised together (mean normalised rank).

    Example requests:
      - Best customers to target while test-mailing only 50 people:
            choose(blind, label, budget=50)
      - Best customers for store visits rather than web orders:
            choose(blind, label, budget=50, goal="NUMSTOREPURCHASES+")
    """
    return _choose(blind_csv, label, budget=budget, goal=goal)

