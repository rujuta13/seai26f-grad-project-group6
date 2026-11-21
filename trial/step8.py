"""Step 8: fix. The fewest input changes that make a bad build good.

Standard library only. Model loading and the kNN predictor are reused from
step2, and the 0-to-1 build score with its training-median threshold from
step6 (the rule used by why). Encoding follows step1: text values get codes
by first appearance in the training rows.

Method: a build is good when its predicted mean goal score is at least the
training median, the same rule as step6.why. Candidate fixes change k
non-frozen input columns to values seen in the training data, for k = 1, 2,
... . The first k with any predicted-good candidate wins, and within that k
the candidate with the highest predicted score wins. The search stops after
MAX_EVALS predictions. If no good candidate is found, the best predicted
build is returned.
"""

import itertools
import statistics

from step2 import _model, _predict
from step6 import _bounds, _scaled


MAX_EVALS = 2000     # predictions allowed in one search
MAX_CHOICES = 8      # values tried per changed column (evenly spaced if more)


def _codes(model):
    """Per input column, the training code of each text value (step1 order:
    by first appearance in the training rows)."""
    tables = {}
    for c in model["inputs"]:
        table = {}
        for r in model["rows"]:
            v = r[c]
            if not isinstance(v, (int, float)):
                table.setdefault(v, len(table))
        tables[c] = table
    return tables


def _encode(inputs, tables, values):
    """One row encoded as step1 encodes it; unseen text gets a fresh code."""
    vec = []
    for c in inputs:
        v = values[c]
        if isinstance(v, (int, float)):
            vec.append(float(v))
        else:
            table = tables[c]
            vec.append(float(table.get(v, len(table))))
    return vec


def _choices(model, base, c):
    """Values column c may take in a fix: training values other than the current one."""
    seen = dict.fromkeys(r[c] for r in model["rows"])
    opts = [v for v in seen if v != base[c]]
    if len(opts) > MAX_CHOICES:
        step = len(opts) / MAX_CHOICES
        opts = [opts[int(i * step)] for i in range(MAX_CHOICES)]
    return opts


def _variants(base, free, choices, k):
    """Every build that differs from base in exactly k free columns."""
    for cols in itertools.combinations(free, k):
        for combo in itertools.product(*(choices[c] for c in cols)):
            values = dict(base)
            values.update(zip(cols, combo))
            yield values


def fix(train_csv, row, frozen=()) -> dict:
    """Inputs of row with as few values changed as possible so the build is
    predicted good. Columns named in frozen keep their values."""
    model = _model(train_csv)
    inputs = model["inputs"]
    goals = model["goals"]
    targets = model["targets"]
    base = {c: row[c] if c in row else 0.0 for c in inputs}
    if not targets or not goals:
        return base

    names = [name for name, _ in goals]
    bounds = _bounds(model)
    train_scores = [
        sum(_scaled(dict(zip(names, t)), goals, bounds)) / len(goals)
        for t in targets
    ]
    median = statistics.median(train_scores)

    tables = _codes(model)
    points = model["points"]

    def score(values):
        pred = _predict(_encode(inputs, tables, values), points, targets)
        return sum(_scaled(dict(zip(names, pred)), goals, bounds)) / len(goals)

    current = score(base)
    if current >= median:
        return base

    frozen_set = set(frozen)
    free = [c for c in inputs if c not in frozen_set]
    choices = {c: _choices(model, base, c) for c in free}

    best_score, best = current, base   # fallback: best prediction seen
    evals = 0
    for k in range(1, len(free) + 1):
        if evals >= MAX_EVALS:
            break
        winner = None
        for values in _variants(base, free, choices, k):
            if evals >= MAX_EVALS:
                break
            evals += 1
            s = score(values)
            if s > best_score:
                best_score, best = s, values
            if s >= median and (winner is None or s > winner[0]):
                winner = (s, values)
        if winner is not None:
            return dict(winner[1])
    return dict(best)

