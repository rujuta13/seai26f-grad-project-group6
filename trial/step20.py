"""Step 20: race. Does our shortlist beat a simple top-overall list? Does our guesser beat an off-the-shelf model?

Runs step10.race on test_csv, which gives the tuning rivals ("ours",
"random_search", "local_search") and the prediction rivals ("ours_guesser",
"linear_model"). Adds the client's simple baseline, "top_overall": scouts
shortlist the _BUDGET test players with the highest OVR rating, then pick the
best of that shortlist on the goals. All scores lie in [0, 1], higher is better.
"""

import step1
import step7
import step10

_OVR = "OVR"  # overall rating column used by the simple top-overall list


def _top_overall_score(test_rows, goals, has_ovr):
    """Normalised goal quality of the best player on the top-OVR shortlist."""
    n = len(test_rows)
    budget = min(step10._BUDGET, n)
    if budget <= 0 or not has_ovr or not goals:
        return 0.0

    def ovr_order(i):
        ovr = step7._num(test_rows[i].get(_OVR))
        # Highest OVR first; rows without a usable OVR go last. Ties go to the lower index.
        return (0, -ovr, i) if ovr is not None else (1, 0.0, i)

    shortlist = sorted(range(n), key=ovr_order)[:budget]
    measured = {i: step10._measure(test_rows, goals, i) for i in shortlist}
    pick = step10._best_measured(measured, goals)

    full = {i: step10._measure(test_rows, goals, i) for i in range(n)}
    totals = step7._totals(full, goals)  # normalised over all test rows
    return float(totals[pick] / len(goals))


def race(train_csv, test_csv) -> dict[str, float]:
    """Score our methods and the standard rivals on test_csv.

    Returns {method: score}, higher is better, all in [0, 1]:
      "ours", "random_search", "local_search", "top_overall"  - shortlist/tuning,
          normalised goal quality of the chosen player
      "ours_guesser", "linear_model"                          - prediction, 1 - MAE / range
    """
    results = dict(step10.race(train_csv, test_csv))

    header, test_rows = step1._read(test_csv)
    inputs, goals_max, goals_min = step1._classify(header)
    goals = [(g, 1.0) for g in goals_max] + [(g, -1.0) for g in goals_min]
    has_ovr = _OVR in header

    if test_rows and goals:
        results["top_overall"] = _top_overall_score(test_rows, goals, has_ovr)
    else:
        results["top_overall"] = 0.0
    results.setdefault("ours", 0.0)

    return {k: float(v) for k, v in results.items()}

