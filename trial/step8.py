"""step8.py - client question 8: fix.

fix(train_csv, row, frozen=()) returns the inputs of a bad build with as few
values changed as possible, so that the build becomes good. Columns named in
`frozen` are never changed.

Reused from earlier steps:
  step2.guess   estimates the goals of a changed flag set, which has not been
                benchmarked, so each candidate can be judged
  step6._model  training percentiles, goal order and the bad/good cut-off
  step6._pct_of percentile of a value on the same scale as step5._percentiles

Search: candidates change 1, 2, ... up to MAX_CHANGES input values. A changed
value is taken from the values seen in the training data for that column.
Sizes are tried in increasing order. The first size that yields a good
candidate wins, and within that size the candidate with the highest mean goal
percentile is returned. If no good candidate is found within MAX_CHANGES
changes or MAX_EVALUATIONS estimates, the highest-scoring candidate tried is
returned, so the result may still be bad.
"""
import itertools

from step2 import guess
from step6 import _model, _pct_of

MAX_CHANGES = 3
MAX_VALUES_PER_COLUMN = 8
MAX_EVALUATIONS = 5000


def _score(M, est):
    """Mean goal percentile of estimated goals, scored as step5/step6 do.
    est maps goal name -> (estimate, low, high) from step2.guess."""
    vec = [sign * est[name][0] for name, sign in M.goal_cols]
    pcts = [_pct_of(v, M.goal_sorted[g]) for g, v in enumerate(vec)]
    return sum(pcts) / len(pcts)


def _domain(M, t):
    """Values column t takes in the training data, thinned to at most
    MAX_VALUES_PER_COLUMN evenly spaced values."""
    vals = sorted({M.X[i][t] for i in range(len(M.X))})
    if len(vals) > MAX_VALUES_PER_COLUMN:
        step = (len(vals) - 1) / (MAX_VALUES_PER_COLUMN - 1)
        vals = [vals[round(k * step)] for k in range(MAX_VALUES_PER_COLUMN)]
    return vals


def _candidates(base, input_cols, options):
    """Yield (k, candidate) for every change of k input values, k = 1, 2, ...
    options[t] lists the new values allowed for input column index t."""
    mutable = sorted(options)
    for k in range(1, len(mutable) + 1):
        for cols in itertools.combinations(mutable, k):
            for picks in itertools.product(*(options[t] for t in cols)):
                cand = dict(base)
                for t, v in zip(cols, picks):
                    cand[input_cols[t]] = v
                yield k, cand


def fix(train_csv, row, frozen=()) -> dict:
    """Inputs of `row` with as few values changed as possible so that the
    build is good. `row` is a bad build with goals removed. Columns named in
    `frozen` are never changed. Returns {input column: value} for every input
    column."""
    M = _model(train_csv)
    input_cols = M.input_cols

    if isinstance(frozen, str):
        frozen = (frozen,)
    frozen = set(frozen)
    unknown = frozen - set(input_cols)
    if unknown:
        raise ValueError(
            f"frozen columns are not input columns: {sorted(unknown)}")

    missing = [c for c in input_cols if c not in row]
    if missing:
        raise KeyError(f"row is missing input columns: {missing}")

    base = {c: float(row[c]) for c in input_cols}

    base_score = _score(M, guess(train_csv, base))
    if base_score > M.cut:
        return base  # already good: no change needed

    # allowed new values per column: other training values, never frozen ones
    options = {}
    for t, c in enumerate(input_cols):
        if c in frozen:
            continue
        opts = [v for v in _domain(M, t) if v != base[c]]
        if opts:
            options[t] = opts

    found = None     # (k, score, candidate): best good candidate at size k
    fallback = (base_score, base)
    tried = 0
    for k, cand in _candidates(base, input_cols, options):
        if found is not None and k > found[0]:
            break  # smaller change sets already gave a good build
        if k > MAX_CHANGES or tried >= MAX_EVALUATIONS:
            break
        s = _score(M, guess(train_csv, cand))
        tried += 1
        if s > M.cut and (found is None or s > found[1]):
            found = (k, s, cand)
        if s > fallback[0]:
            fallback = (s, cand)

    if found is not None:
        return found[2]
    return fallback[1]

