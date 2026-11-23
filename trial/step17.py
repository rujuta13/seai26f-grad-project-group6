from step7 import choose as _choose


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Return the index of the best row found, labelling at most `budget` rows.

    blind_csv: CSV whose goal cells are all '?'. Rows the client rules out
               (for example, age or league limits) should be removed from this
               file before calling, so they are never labelled.
    label(i):  returns a dict {goal column: value} for row i. Called at most
               `budget` times; a further call raises RuntimeError.
    goal:      optional name of one goal column to optimise, for example
               "Penalties+". If None, all goal columns are optimised together.

    Example requests:
      - Best strong penalty-taker with 20 labels:
            choose(blind, label, budget=20, goal="Penalties+")
      - Best player under 24 in a scouted league, with the default budget:
            choose(blind_filtered, label, budget=30)
    """
    calls = 0

    def counted_label(i):
        nonlocal calls
        if calls >= budget:
            raise RuntimeError("labelling budget exhausted")
        calls += 1
        return label(i)

    return _choose(blind_csv, counted_label, budget=budget, goal=goal)

