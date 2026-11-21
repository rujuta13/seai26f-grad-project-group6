"""step20.py: race our shortlist and penalty guesser against simple rivals.

race(train_csv, test_csv) returns {method: score}. Higher is better for every key.

Shortlist race. test_csv is the pool of candidate players, with goals hidden
until a player is scouted. Each method may scout at most SHORTLIST_BUDGET players.
  - ours: step17.choose (step7's search), averaged over SEEDS.
  - top_rating: the client's "top overall rating" list. Scouts go down the
    players sorted by OVR, highest first, and keep the best one seen.
  The score of a chosen player is its goal quality over the test pool (step10's
  normalised goal sum, 1 = best on every goal).

Penalty guesser race. Models are trained on train_csv and tested on test_csv
for the Penalties+ goal only.
  - penalty_guess_ours: step12.guess, the calibrated nearest-neighbour median.
  - penalty_guess_linear: ordinary least squares on the numeric attributes, an
    off-the-shelf model. Missing attributes are filled with training means.
  The score is R^2 on test_csv (1 = perfect, 0 = no better than the test mean).

Reuses step7 and step17 (search), step10 (benchmark wrapper, quality, least
squares, R^2), step12 (guesser) and step1 (loading). Standard library only.
"""
import math
import os
import tempfile

import step7
import step10
from step1 import _load, _columns
from step12 import guess
from step17 import choose

SHORTLIST_BUDGET = 30        # players the scouts may benchmark per method
SEEDS = (1, 2, 3, 4, 5)      # search seeds for our method (the rival is deterministic)
RATING_COL = "OVR"           # the client's "overall rating"
PENALTY = "Penalties"        # goal name without its +/- suffix


def _ours(blind, label, seed, budget):
    """step17.choose (step7's search) with its random seed set to `seed`."""
    old = step7.SEED
    step7.SEED = seed
    try:
        return choose(blind, label, budget=budget)
    finally:
        step7.SEED = old


def _rating(row):
    """The overall rating of a row, or -inf if it is missing or not finite."""
    v = row.get(RATING_COL)
    if isinstance(v, float) and math.isfinite(v):
        return v
    return -math.inf


def _race_shortlist(cols, rows, goals, budget):
    names = [c for c, _ in goals]
    g = len(goals)
    q = step10._quality([step10._oriented(r, goals) for r in rows], g)

    def label(i):
        return {c: rows[i][c] for c in names}

    # Ours: average the chosen player's quality over the seeds.
    ours_total = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        blind = os.path.join(tmp, "blind.csv")
        step10._write_blind(cols, rows, set(names), blind)
        for seed in SEEDS:
            ours_total += q[_ours(blind, label, seed, budget)]

    # Rival: scout down the list sorted by rating, highest first, and keep the
    # best player seen.
    order = sorted(range(len(rows)), key=lambda i: (-_rating(rows[i]), i))
    bench = step10._Bench(label, goals, budget)
    for i in order[:budget]:
        bench(i)
    top = step10._chosen_quality(bench.seen, q, g)

    return {"ours": ours_total / len(SEEDS), "top_rating": top}


def _penalty_goal(goal_names):
    """The name of the penalty goal column, or None if there is none."""
    for c in goal_names:
        if c == PENALTY or c[:-1] == PENALTY:
            return c
    return None


def _race_penalty(train_csv, rows, goal):
    tcols, trows = _load(train_csv)
    inputs, maxi, mini = _columns(tcols, trows)
    if goal not in set(maxi + mini):
        raise ValueError(f"{train_csv} has no {goal} goal column")
    if not trows:
        raise ValueError(f"{train_csv} has no rows")

    # Training means fill attributes missing from a test row for the linear model.
    means = {c: sum(r[c] for r in trows) / len(trows) for c in inputs}

    pts = [([1.0] + [float(r[c]) for c in inputs], float(r[goal]))
           for r in trows
           if isinstance(r[goal], float) and math.isfinite(r[goal])]
    if not pts:
        raise ValueError(f"{train_csv} has no finite {goal} values")
    w = step10._fit_one([x for x, _ in pts], [y for _, y in pts])

    ours_pairs, lin_pairs = [], []
    for r in rows:
        y = r.get(goal)
        if not (isinstance(y, float) and math.isfinite(y)):
            continue
        flags = {c: r.get(c) for c in inputs}
        est = guess(train_csv, flags)[goal][0]
        x = [1.0] + [float(r[c]) if isinstance(r.get(c), float) else means[c]
                     for c in inputs]
        ours_pairs.append((y, est))
        lin_pairs.append((y, step10._predict(w, x)))

    return {"penalty_guess_ours": step10._r2(ours_pairs),
            "penalty_guess_linear": step10._r2(lin_pairs)}


def race(train_csv, test_csv) -> dict[str, float]:
    """Score our shortlist ("ours") against the top-rating list ("top_rating"),
    and our penalty guesser against an off-the-shelf linear model on Penalties+.

    Shortlist scores lie in [0, 1]; guesser scores are R^2 (at most 1).
    Higher is better for every key. If test_csv has no penalty goal, only the
    shortlist keys are returned.
    """
    cols, rows = _load(test_csv)
    _, maxi, mini = _columns(cols, rows)
    goals = [(c, 1.0) for c in maxi] + [(c, -1.0) for c in mini]
    if not goals or not rows:
        raise ValueError(f"{test_csv} needs goal columns and rows")

    budget = min(SHORTLIST_BUDGET, len(rows))
    out = _race_shortlist(cols, rows, goals, budget)

    penalty = _penalty_goal([c for c, _ in goals])
    if penalty is not None:
        out.update(_race_penalty(train_csv, rows, penalty))

    return {k: float(v) for k, v in out.items()}

