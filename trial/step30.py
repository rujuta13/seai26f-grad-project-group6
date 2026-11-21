"""step30.py: race our targeting and response guesser against simple rivals.

race(train_csv, test_csv) returns {method: score}. Higher is better for every key.

Targeting race. test_csv is the pool of customers, with goals hidden until a
customer is mailed. Each method may mail at most MAILING_BUDGET customers.
  - ours: step27.choose (step7's search), averaged over SEEDS.
  - highest_income: the client's rival. Mail customers in order of INCOME,
    highest first, and keep the best one seen.
  The score is the chosen customer's goal quality over the test pool (step10's
  normalised goal sum, 1 = best on every goal).

Response guesser race. Models are trained on train_csv and tested on test_csv
for RESPONSE+ only.
  - response_guess_ours: step22.guess, the calibrated nearest-neighbour median
    clipped to the training range.
  - response_guess_linear: ordinary least squares on the numeric flags, an
    off-the-shelf model (step10's solver), with training means filling gaps.
  The score is R^2 on test_csv (1 = perfect, 0 = no better than the test mean).

Reuses step7 (search seed), step27 (targeting), step22 (guesser), step10
(benchmark wrapper, quality, least squares, R^2) and step1 (loading).
Standard library only.
"""
import math
import os
import tempfile

import step7
import step10
from step1 import _load, _columns
from step22 import guess as _guess_response
from step27 import choose as _choose

MAILING_BUDGET = 50      # customers the team may mail per method
SEEDS = (1, 2, 3, 4, 5)  # search seeds for our method (the rival is deterministic)
INCOME_COL = "INCOME"    # the client's rival ranks customers by this column
RESPONSE = "RESPONSE"    # goal name without its +/- suffix


def _income(row):
    """The income of a row, or -inf if it is missing or not finite."""
    v = row.get(INCOME_COL)
    if isinstance(v, float) and math.isfinite(v):
        return v
    return -math.inf


def _response_goal(goal_names):
    """The name of the response goal column, or None if there is none."""
    for c in goal_names:
        if c == RESPONSE or c[:-1] == RESPONSE:
            return c
    return None


def _race_targeting(cols, rows, goals, budget):
    names = [c for c, _ in goals]
    g = len(goals)
    q = step10._quality([step10._oriented(r, goals) for r in rows], g)

    def label(i):
        return {c: rows[i][c] for c in names}

    # Ours: average the chosen customer's quality over the seeds.
    ours_total = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        blind = os.path.join(tmp, "blind.csv")
        step10._write_blind(cols, rows, set(names), blind)
        for seed in SEEDS:
            old = step7.SEED
            step7.SEED = seed
            try:
                pick = _choose(blind, label, budget=budget)
            finally:
                step7.SEED = old
            ours_total += q[pick]

    # Rival: mail the highest-income customers first, and keep the best seen.
    order = sorted(range(len(rows)), key=lambda i: (-_income(rows[i]), i))
    bench = step10._Bench(label, goals, budget)
    for i in order[:budget]:
        bench(i)
    rich = step10._chosen_quality(bench.seen, q, g)

    return {"ours": ours_total / len(SEEDS), "highest_income": rich}


def _race_response(train_csv, rows, goal):
    tcols, trows = _load(train_csv)
    inputs, maxi, mini = _columns(tcols, trows)
    if goal not in set(maxi + mini) or not trows:
        return {}

    # Training means fill attributes missing from a test row for the linear model.
    means = {c: sum(r[c] for r in trows) / len(trows) for c in inputs}

    pts = [([1.0] + [float(r[c]) for c in inputs], float(r[goal]))
           for r in trows
           if isinstance(r[goal], float) and math.isfinite(r[goal])]
    if not pts:
        return {}
    w = step10._fit_one([x for x, _ in pts], [y for _, y in pts])

    ours_pairs, lin_pairs = [], []
    for r in rows:
        y = r.get(goal)
        if not (isinstance(y, float) and math.isfinite(y)):
            continue
        flags = {c: r.get(c) for c in inputs}
        est = _guess_response(train_csv, flags)[goal][0]
        x = [1.0] + [float(r[c]) if isinstance(r.get(c), float) else means[c]
                     for c in inputs]
        ours_pairs.append((y, est))
        lin_pairs.append((y, step10._predict(w, x)))

    return {"response_guess_ours": step10._r2(ours_pairs),
            "response_guess_linear": step10._r2(lin_pairs)}


def race(train_csv, test_csv) -> dict[str, float]:
    """Score our targeting ("ours") against the highest-income rival
    ("highest_income"), and our response guesser against an off-the-shelf
    linear model on RESPONSE+.

    Targeting scores lie in [0, 1]; guesser scores are R^2 (at most 1).
    Higher is better for every key. If test_csv has no response goal, only
    the targeting keys are returned.
    """
    cols, rows = _load(test_csv)
    _, maxi, mini = _columns(cols, rows)
    goals = [(c, 1.0) for c in maxi] + [(c, -1.0) for c in mini]
    if not goals or not rows:
        raise ValueError(f"{test_csv} needs goal columns and rows")

    budget = min(MAILING_BUDGET, len(rows))
    out = _race_targeting(cols, rows, goals, budget)

    response = _response_goal([c for c, _ in goals])
    if response is not None:
        out.update(_race_response(train_csv, rows, response))

    return {k: float(v) for k, v in out.items()}

