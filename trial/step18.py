"""step18.py: the fewest attribute changes that make a player good.

fix(train_csv, row, frozen=()) answers both client questions:
  - "This player is close: what would he need to improve to make our
    shortlist?" Pass his attributes as row, and the attributes he cannot
    change (age, nation, ...) as frozen.
  - "What training focus would turn our weak penalty-taker into a reliable
    one?" The same call. The changed attributes are the training focus.

The search is step8's: all single changes first, then pairs, then triples,
and the first size that yields a predicted-good player wins.

Differences from step8.fix:
  - Predictions come from step12.guess, the calibrated nearest-neighbour
    model that step16.why uses, so both steps judge players the same way.
  - The goodness test checks only the Pareto front of the training players.
    Dominance is transitive, so a player is beaten by some training player
    exactly when a front player beats him. This is much faster.

Reuses step1 loading and column roles, step8's search limits and value
thinning, and step11's fast front. Standard library only.
"""
import math

from step1 import _load, _columns, _dominates
from step8 import MAX_CHANGES, MAX_VALUES, MAX_EVALS, _thin, _changesets
from step11 import _pareto_front
from step12 import guess


def _finite(v):
    return isinstance(v, float) and math.isfinite(v)


def fix(train_csv, row, frozen=()) -> dict:
    """Return row's inputs with as few values changed as possible so the player
    becomes good. Columns named in `frozen` are never changed.

    The row is returned unchanged if it is already good or if nothing can be
    changed. If no predicted-good player is found within the search limits,
    the best-scoring change found is returned.
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
             if all(_finite(r[c]) for c, _ in goals)]
    if not train:
        return result
    front = [train[i] for i in _pareto_front(train)]

    def is_good(v):
        # v is beaten by some training player exactly when it is beaten by a
        # player on the front, because dominance is transitive.
        return not any(_dominates(t, v) for t in front)

    frozen = set(frozen)
    mutable = [c for c in inputs if c not in frozen]

    # Candidate new values per flag: its finite training values, thinned,
    # minus the current one.
    options = {}
    for c in mutable:
        current = row.get(c)
        values = sorted({r[c] for r in rows if _finite(r[c])})
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
        """Oriented goal vector predicted from a flag set by step12.guess."""
        p = guess(train_csv, flags)
        return [sgn * p[c][0] for c, sgn in goals]

    v0 = predict(row)
    if is_good(v0):
        return result

    # Fallback if nothing good is found: the best-scoring change seen so far.
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

