"""step20.py: race, for the two client questions.

Q1: "Does our shortlist beat a simple 'top overall rating' list?"
Q2: "Does our penalty guesser beat an off-the-shelf model?"

Q1: our shortlist is step7.choose over all goals together (here Penalties+ and
Strength+). The simple list is the BUDGET test rows with the highest OVR,
measured, with the best of them taken by the same leader rule as step7.
Random search from step10 is a further rival.

Q2: our penalty guesser is step12.guess read at Penalties+ alone. The
off-the-shelf stand-in is the ridge regression from step10. Both are scored
by R^2 on Penalties+.

Score scales: the shortlist keys score the chosen build's true goals on [0, 1]
(higher is better). The guesser keys score R^2 on Penalties+ (higher is
better, can be negative). Compare within each group, not across them.

Reading, column splitting, number parsing, goal vectors, normalisation, the
leader rule, the tuner, the ridge rival, the R^2 scorer and the guesser all
come from steps 1, 4, 5, 7, 10 and 12. Only the OVR list and the wiring are
new here.
"""
from __future__ import annotations

import math

from step1 import _columns, _num, _read
from step4 import _objectives
from step5 import _scores
from step7 import _leader, _resolve_goal, _signed, choose
from step10 import BUDGET, _goal_values, _linear_preds, _model_score, _random_search
from step12 import guess as _guess

OVR_COLUMN = "OVR"          # the simple list ranks players by this input column
PENALTY = "Penalties"       # base name of the penalty goal (Penalties+)


def _ovr(row: dict) -> float:
    """OVR as a finite float, or -inf if it is missing or unusable."""
    v = _num(row.get(OVR_COLUMN))
    return v if v is not None else -math.inf


def _top_overall(rows: list[dict], goals: list[str], sign: list[float],
                 budget: int) -> int:
    """The simple list: measure the `budget` rows with the highest OVR (ties keep
    file order) and return the leader among them by the step7 rule."""
    order = sorted(range(len(rows)), key=lambda i: (-_ovr(rows[i]), i))
    obs = {}
    for i in order[:budget]:
        obs[i] = _signed(_goal_values(rows, i, goals), goals, sign)
    return _leader(obs, len(goals))


def _shortlist(test_csv, test: list[dict], goals: list[str], budget: int) -> int:
    """Our shortlist: step7.choose over all goals together, labelled from test."""
    def label(i: int) -> dict:
        return _goal_values(test, i, goals)

    return choose(test_csv, label, budget=budget)


def _penalty_goal(goals: list[str]) -> str | None:
    """The Penalties goal column (e.g. 'Penalties+'), or None if absent."""
    try:
        return _resolve_goal(PENALTY, goals)
    except ValueError:
        return None


def _penalty_race(train_csv, train: list[dict], inputs: list[str],
                  test: list[dict], pen: str) -> dict[str, float]:
    """Guessers scored by R^2 on the Penalties goal alone."""
    truth = {pen: [_num(r.get(pen)) for r in test]}

    train_vals = [v for v in (_num(r.get(pen)) for r in train) if v is not None]
    fallback = {pen: sum(train_vals) / len(train_vals) if train_vals else 0.0}

    # Our guesser: step12.guess, one call per test row.
    preds = []
    for row in test:
        est = _guess(train_csv, row)
        preds.append(float(est.get(pen, (math.nan, 0.0, 0.0))[0]))

    # Off-the-shelf stand-in: ridge regression on encoded inputs.
    linear = _linear_preds(train, inputs, test, [pen])

    return {
        "ours_guesser": float(_model_score(truth, {pen: preds}, fallback)),
        "linear_model": float(_model_score(truth, linear, fallback)),
    }


def race(train_csv, test_csv) -> dict[str, float]:
    """Race the methods on test_csv.

    Shortlist keys (score in [0, 1], higher is better; the mean normalised goal
    score of the build each method picks, using its true test goals):
      "ours"                step7.choose with BUDGET label calls (our shortlist)
      "top_overall_rating"  the BUDGET rows with the highest OVR, same leader rule
      "random_search"       random sample of BUDGET builds, same leader rule

    Guesser keys (R^2 on Penalties+, higher is better, can be negative):
      "ours_guesser"        step12.guess estimates for Penalties+, one call per row
      "linear_model"        stdlib stand-in for an off-the-shelf model (ridge)
    """
    header, test = _read(str(test_csv))
    _, gmax, gmin = _columns(header)
    goals = gmax + gmin
    names = ["ours", "top_overall_rating", "random_search",
             "ours_guesser", "linear_model"]
    n = len(test)
    if n == 0 or not goals:
        return {k: 0.0 for k in names}

    train_header, train = _read(str(train_csv))
    inputs, _, _ = _columns(train_header)

    sign = [1.0 if g in gmax else -1.0 for g in goals]
    obj = [_objectives(r, gmax, gmin) for r in test]
    quality = _scores(obj, len(goals))
    budget = min(BUDGET, n)

    picks = {
        "ours": _shortlist(test_csv, test, goals, budget),
        "top_overall_rating": _top_overall(test, goals, sign, budget),
        "random_search": _random_search(test, goals, sign, budget),
    }
    out: dict[str, float] = {k: float(quality[i]) for k, i in picks.items()}

    pen = _penalty_goal(goals)
    if pen is None:
        out["ours_guesser"] = 0.0
        out["linear_model"] = 0.0
    else:
        out.update(_penalty_race(train_csv, train, inputs, test, pen))

    return out

