"""step30.py: race, for the two client questions.

Q1: "Does our targeting beat simply mailing our highest-income customers?"
Q2: "Does our response guesser beat an off-the-shelf model?"

Q1 compares step7.choose (all goals together, BUDGET label calls) with the rule
"mail the BUDGET customers with the highest INCOME", judged by the step7 leader
rule over the measured customers. Random search from step10 is a further rival.
Scores are the mean normalised goal score (in [0, 1], higher is better) of the
chosen customer, using true test goals.

Q2 compares the RESPONSE+ guesses of step12 (via step22's neighbour rule) with
the ridge regression stand-in from step10. Scores are R^2 on RESPONSE+ (higher
is better, can be negative). Compare within each group, not across them.

Reading, column splitting, goal vectors, normalisation, the leader rule, the
tuner, the random rival, the ridge rival and the R^2 scorer come from steps
1, 4, 5, 7, 10 and 12. Only the income list and the wiring are new here.
"""
from __future__ import annotations

import math

from step1 import _columns, _num, _read
from step4 import _objectives
from step5 import _scores
from step7 import _leader, _resolve_goal, _signed, choose
from step10 import BUDGET, _goal_values, _linear_preds, _model_score, _random_search
from step12 import guess as _guess

INCOME_COLUMN = "INCOME"    # the simple rival mails the customers with the highest value here
RESPONSE = "RESPONSE"       # base name of the campaign response goal (RESPONSE+)


def _income(row: dict) -> float:
    """INCOME as a finite float, or -inf if it is missing or unusable."""
    v = _num(row.get(INCOME_COLUMN))
    return v if v is not None else -math.inf


def _top_income(rows: list[dict], goals: list[str], sign: list[float],
                budget: int) -> int:
    """The simple rival: measure the `budget` rows with the highest INCOME (ties
    keep file order) and return the leader among them by the step7 rule."""
    order = sorted(range(len(rows)), key=lambda i: (-_income(rows[i]), i))
    obs = {}
    for i in order[:budget]:
        obs[i] = _signed(_goal_values(rows, i, goals), goals, sign)
    return _leader(obs, len(goals))


def _targeting(test_csv, test: list[dict], goals: list[str], budget: int) -> int:
    """Our targeting: step7.choose over all goals together, labelled from test."""
    def label(i: int) -> dict:
        return _goal_values(test, i, goals)

    return choose(test_csv, label, budget=budget)


def _response_goal(goals: list[str]) -> str | None:
    """The RESPONSE goal column (e.g. 'RESPONSE+'), or None if absent."""
    try:
        return _resolve_goal(RESPONSE, goals)
    except ValueError:
        return None


def _response_race(train_csv, train: list[dict], inputs: list[str],
                   test: list[dict], resp: str) -> dict[str, float]:
    """Guessers scored by R^2 on the RESPONSE goal alone."""
    truth = {resp: [_num(r.get(resp)) for r in test]}

    train_vals = [v for v in (_num(r.get(resp)) for r in train) if v is not None]
    fallback = {resp: sum(train_vals) / len(train_vals) if train_vals else 0.0}

    # Our guesser: step12.guess (neighbour estimate), one call per test row.
    preds = []
    for row in test:
        est = _guess(train_csv, row)
        preds.append(float(est.get(resp, (math.nan, 0.0, 0.0))[0]))

    # Off-the-shelf stand-in: ridge regression on encoded inputs.
    linear = _linear_preds(train, inputs, test, [resp])

    return {
        "ours_guesser": float(_model_score(truth, {resp: preds}, fallback)),
        "linear_model": float(_model_score(truth, linear, fallback)),
    }


def race(train_csv, test_csv) -> dict[str, float]:
    """Race the methods on test_csv.

    Targeting keys (score in [0, 1], higher is better; the mean normalised goal
    score of the customer each method picks, using its true test goals):
      "ours"            step7.choose with BUDGET label calls (our targeting)
      "highest_income"  the BUDGET customers with the highest INCOME, same leader rule
      "random_search"   random sample of BUDGET customers, same leader rule

    Guesser keys (R^2 on RESPONSE+, higher is better, can be negative):
      "ours_guesser"    step12.guess estimates for RESPONSE+, one call per row
      "linear_model"    stdlib stand-in for an off-the-shelf model (ridge)
    """
    header, test = _read(str(test_csv))
    _, gmax, gmin = _columns(header)
    goals = gmax + gmin
    names = ["ours", "highest_income", "random_search",
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
        "ours": _targeting(test_csv, test, goals, budget),
        "highest_income": _top_income(test, goals, sign, budget),
        "random_search": _random_search(test, goals, sign, budget),
    }
    out: dict[str, float] = {k: float(quality[i]) for k, i in picks.items()}

    resp = _response_goal(goals)
    if resp is None:
        out["ours_guesser"] = 0.0
        out["linear_model"] = 0.0
    else:
        out.update(_response_race(train_csv, train, inputs, test, resp))

    return out

