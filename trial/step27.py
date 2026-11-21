import step7


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best customer found, labelling at most `budget` customers.

    blind_csv has every goal cell set to '?'. label(i) returns the goals of
    customer row i as a dict and may be called at most `budget` times; a
    further call raises RuntimeError. If `goal` names one goal column (for
    example "NUMSTOREPURCHASES+" or "NUMSTOREPURCHASES"), only that goal is
    optimised. For the two client questions, pass budget=50 for the test
    mailing, and pass goal="NUMSTOREPURCHASES+" to target store visits
    instead of web orders.

    The search itself is step7.choose. Customers are not filtered here, so
    the caller should pass a blind_csv that already holds only the eligible
    customers.
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

