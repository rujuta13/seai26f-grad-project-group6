import math

import step12


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate each goal (e.g. web purchases, store purchases, campaign response)
    for a new customer whose goal columns are not known.

    Delegates to step12.guess, which uses the nearest training customers. Each
    goal gets a weighted-median estimate and a 5th-95th percentile range. The
    estimate is kept inside its range, and non-finite values fall back to 0.0.
    """
    out = {}
    for name, triple in step12.guess(train_csv, row).items():
        est, lo, hi = (float(x) for x in triple)
        if not math.isfinite(est):
            est = 0.0
        if math.isnan(lo):
            lo = -math.inf
        if math.isnan(hi):
            hi = math.inf
        lo = min(lo, est)
        hi = max(hi, est)
        out[name] = (est, lo, hi)
    return out

