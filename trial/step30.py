"""Step 30: race. Our targeting and response guesser against standard rivals on test data.

Serves the client's questions:
  - "Does our targeting beat simply mailing our highest-income customers?"
  - "Does our response guesser beat an off-the-shelf model?"

Builds on step20.race, which scores our tuner and guesser against random search,
local search, a top-OVR shortlist and a ridge regression. This module adds the
client's own baseline, "top_income": test-mail the _BUDGET test customers with
the highest INCOME, then pick the best of that list on the goals. All scores lie
in [0, 1]; higher is better.
"""

import step1
import step7
import step10
import step20

_INCOME = "INCOME"  # income column used by the simple top-income list


def _top_income_score(test_rows, goals, has_income):
    """Normalised goal quality of the best customer on the top-income shortlist."""
    n = len(test_rows)
    budget = min(step10._BUDGET, n)
    if budget <= 0 or not has_income or not goals:
        return 0.0

    def income_order(i):
        inc = step7._num(test_rows[i].get(_INCOME))
        # Highest income first; rows without a usable income go last.
        # Ties go to the lower index.
        return (0, -inc, i) if inc is not None else (1, 0.0, i)

    shortlist = sorted(range(n), key=income_order)[:budget]
    measured = {i: step10._measure(test_rows, goals, i) for i in shortlist}
    pick = step10._best_measured(measured, goals)

    full = {i: step10._measure(test_rows, goals, i) for i in range(n)}
    totals = step7._totals(full, goals)  # normalised over all test rows
    return float(totals[pick] / len(goals))


def race(train_csv, test_csv) -> dict[str, float]:
    """Score our methods and the standard rivals on test_csv.

    Returns {method: score}, higher is better, all in [0, 1]:
      "ours", "random_search", "local_search", "top_overall", "top_income"
          - targeting: normalised goal quality of the chosen customer
      "ours_guesser", "linear_model"
          - response prediction: 1 - MAE / range
    """
    results = dict(step20.race(train_csv, test_csv))

    header, test_rows = step1._read(test_csv)
    _, goals_max, goals_min = step1._classify(header)
    goals = [(g, 1.0) for g in goals_max] + [(g, -1.0) for g in goals_min]
    has_income = _INCOME in header

    if test_rows and goals:
        results["top_income"] = _top_income_score(test_rows, goals, has_income)
    else:
        results["top_income"] = 0.0
    results.setdefault("ours", 0.0)

    return {k: float(v) for k, v in results.items()}

