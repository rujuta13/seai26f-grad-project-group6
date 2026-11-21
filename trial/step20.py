"""Step 20: race our shortlist and penalty guesser against simple rivals.

race() runs step10's tuner and guesser comparison on test_csv. It also adds a
"top_ovr" rival for the shortlist question: pick the test build with the highest
OVR (ties go to the earlier row) and score it by regret, as step10 does. Every
score is lower-is-better, and "ours" is always present.
"""

import math

from step1 import _load, _split_columns
from step10 import _normalised_scores
from step10 import race as _race_step10


def _finite(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _top_ovr_regret(test_csv):
    """Regret of the test build with the highest OVR, or 1.0 if no build qualifies."""
    columns, test = _load(test_csv)
    maximise, minimise, _ = _split_columns(columns)
    if "OVR" not in columns:
        return 1.0

    scores = _normalised_scores(test, maximise, minimise)
    valid = [i for i, s in enumerate(scores) if s is not None]
    if not valid:
        return 1.0
    best_score = min(scores[i] for i in valid)

    candidates = [i for i in valid if _finite(test[i].get("OVR")) is not None]
    if not candidates:
        return 1.0
    pick = min(candidates, key=lambda i: (-_finite(test[i].get("OVR")), i))
    return float(scores[pick] - best_score)


def race(train_csv, test_csv) -> dict[str, float]:
    """Return {method: score} on test_csv, lower-is-better, including "ours".

    Keys from step10: "ours" and the tuner rivals "random_search" and
    "local_search" (regret), and "guesser_ours_nmae" and "guesser_ridge_nmae"
    (error). Also "top_ovr": the regret of the highest-OVR build, the simple
    top-rating shortlist.
    """
    result = {k: float(v) for k, v in _race_step10(train_csv, test_csv).items()}
    result["top_ovr"] = float(_top_ovr_regret(test_csv))
    return result

