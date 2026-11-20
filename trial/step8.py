"""step8.py: fix a bad build with as few flag changes as possible.

fix(train_csv, row, frozen): a build is good when no training build is at least
as good on every goal and better on one (the step6.why test). The goals of a
changed build are predicted from its flags with step2.guess, the median goal
value of the nearest training flag sets. Changes are searched by size: all
single-flag changes first, then pairs, up to MAX_CHANGES. The first size that
yields a predicted-good build wins, and among those the build with the highest
predicted normalised goal sum is returned. Flags named in `frozen` never change.
If no predicted-good build is found, the best-scoring build found is returned.

Standard library only.
"""
from itertools import combinations, product

from step1 import _load, _columns, _dominates
from step2 import guess

MAX_CHANGES = 3   # largest number of flag values changed
MAX_VALUES = 10   # candidate values per flag: its training values, thinned to this many
MAX_EVALS = 4000  # cap on predictions per call, to bound run time


def _thin(values, limit):
    """At most `limit` values, spread evenly over the sorted list."""
    if len(values) <= limit:
        return values
    step = (len(values) - 1) / (limit - 1)
    return [values[round(i * step)] for i in range(limit)]


def _changesets(mutable, options, k):
    """Every way to change exactly k flags, each to one of its candidate values."""
    for cols_k in combinations(mutable, k):
        for vals in product(*(options[c] for c in cols_k)):
            yield dict(zip(cols_k, vals))


def fix(train_csv, row, frozen=()) -> dict:
    """Return row's inputs with as few values changed as possible so the build
    becomes good. Columns named in `frozen` are never changed.

    Example: fix(csv, row, frozen=("inline",)) answers "the best fix without
    touching inline". The row is returned unchanged if it is already good or if
    nothing can be changed.
    """
    cols, rows = _load(train_csv)
    inputs, maxi, mini = _columns(cols, rows)
    goals = [(c, 1.0) for c in maxi] + [(c, -1.0) for c in mini]
    result = dict(row)
    if not goals or not rows:
        return result

    # Training goal vectors, oriented so higher is better.
    train = [[sgn * r[c] for c, sgn in goals]
             for r in rows
             if all(isinstance(r[c], float) for c, _ in goals)]
    if not train:
        return result

    frozen = set(frozen)
    mutable = [c for c in inputs if c not in frozen]

    # Candidate new values per flag: its training values, thinned, minus the current one.
    options = {}
    for c in mutable:
        current = row.get(c)
        values = sorted({r[c] for r in rows})
        options[c] = [v for v in _thin(values, MAX_VALUES) if v != current]

    g = len(goals)
    lo = [min(t[k] for t in train) for k in range(g)]
    hi = [max(t[k] for t in train) for k in range(g)]

    def score(v):
        """Sum over goals of the position between the worst and best training value."""
        s = 0.0
        for k in range(g):
            span = hi[k] - lo[k]
            s += (v[k] - lo[k]) / span if span > 0 else 1.0
        return s

    def predict(flags):
        """Oriented goal vector predicted from a flag set by step2.guess."""
        p = guess(train_csv, flags)
        return [sgn * p[c][0] for c, sgn in goals]

    def is_good(v):
        return not any(_dominates(t, v) for t in train)

    v0 = predict(row)
    if is_good(v0):
        return result

    # Fallback if nothing good is found: the best-scoring build seen so far.
    fallback = (score(v0), {})
    evals = 0
    for k in range(1, min(MAX_CHANGES, len(mutable)) + 1):
        found = []
        for changes in _changesets(mutable, options, k):
            if evals >= MAX_EVALS:
                break
            evals += 1
            flags = dict(row)
            flags.update(changes)
            v = predict(flags)
            s = score(v)
            if is_good(v):
                found.append((s, changes))
            elif s > fallback[0]:
                fallback = (s, changes)
        if found:
            _, changes = max(found, key=lambda t: t[0])
            result.update(changes)
            return result
        if evals >= MAX_EVALS:
            break

    result.update(fallback[1])
    return result

