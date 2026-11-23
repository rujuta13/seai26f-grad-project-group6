from step2 import guess as _guess_base


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for a player whose goal columns are unknown.

    Returns {goal: (estimate, low, high)}. The estimate comes from a random
    forest fitted on the training rows, and low..high is the 10th to 90th
    percentile range of that model's held-out errors, shifted onto the
    estimate. Goal cells in `row` are ignored, since only the input columns
    are used.
    """
    est = _guess_base(train_csv, row)
    return {g: (float(mid), float(lo), float(hi))
            for g, (mid, lo, hi) in est.items()}

