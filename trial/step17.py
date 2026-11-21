import step7

_SEED = 0  # kept for parity with the earlier steps; step7 fixes its own seed


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found, labelling at most `budget` rows.

    blind_csv has every goal cell set to '?'. label(i) returns the goals of
    row i as a dict and may be called at most `budget` times; a further call
    raises RuntimeError. If `goal` names one goal column (for example
    "Penalties+" or "Penalties"), only that goal is optimised.

    The search itself is step7.choose. Restrictions on which players are
    candidates (for example age under 24, or a league already scouted) are
    not part of this function, so the caller should pass a blind_csv that
    already contains only the eligible rows.
    """
    if not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a positive integer")

    calls = [0]

    def counted(i):
        # guard the budget here as well, so the limit holds regardless of
        # how the underlying search asks for labels
        if calls[0] >= budget:
            raise RuntimeError(f"label budget of {budget} exhausted")
        calls[0] += 1
        return label(i)

    return int(step7.choose(blind_csv, counted, budget=budget, goal=goal))

