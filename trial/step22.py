import numpy as np

from step1 import _load
from step2 import _columns, guess as _guess_base


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate each goal for a customer whose goal columns are removed.

    Returns {goal: (estimate, low, high)} for every goal. The estimate and the
    interval come from step2.guess, a random forest with held-out-error
    calibration. Results are clipped to the range of values seen in the
    training data, so counts stay non-negative and RESPONSE stays within 0..1.
    """
    est = _guess_base(train_csv, row)
    if not est:
        return {}

    _, goals = _columns(train_csv)
    rows = _load(train_csv)["rows"]

    out = {}
    for g in goals:
        if g not in est:
            continue
        mid, lo, hi = est[g]
        vals = np.array([float(r[g]) for r in rows], dtype=float)
        gmin, gmax = float(vals.min()), float(vals.max())

        mid = min(max(float(mid), gmin), gmax)
        lo = min(max(float(lo), gmin), gmax)
        hi = min(max(float(hi), gmin), gmax)
        lo, hi = min(lo, mid), max(hi, mid)
        out[g] = (mid, lo, hi)
    return out

