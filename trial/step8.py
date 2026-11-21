"""Step 8: fix a slow build by changing as few flags as possible.

fix() takes a bad build (flag settings only, goals removed) and searches for the
smallest number of flag changes that the training data predicts will make it good.
Candidate builds are scored with step2's guess(). "Good" follows step6's why():
the normalised goal sum is no worse than the 75th percentile of the training
builds. Columns named in `frozen` are never changed. The search tries 1 change,
then 2, and so on. Among the candidates with the fewest changes it returns the
one with the best predicted goal sum. Every value a column takes is one already
seen in the training data. If no good build is found within the search budget,
the predicted-best build found is returned.
"""

import itertools
import math

from step1 import _load, _split_columns
from step2 import _quantile, guess

MAX_EVALS = 2000
GOOD_QUANTILE = 0.75


def _as_finite(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _vector(values, maximise, minimise):
    """Minimisation vector for goal values ('+' negated), or None if any is missing."""
    vec = []
    for c in maximise:
        v = _as_finite(values.get(c))
        if v is None:
            return None
        vec.append(-v)
    for c in minimise:
        v = _as_finite(values.get(c))
        if v is None:
            return None
        vec.append(v)
    return vec


def _domains(rows, columns):
    """Per column: the distinct training values, in first-seen order."""
    domain = {}
    for c in columns:
        seen = {}
        for r in rows:
            v = r.get(c)
            if v is None:
                continue
            seen.setdefault(str(v), v)
        domain[c] = list(seen.values())
    return domain


def _changes(changeable, base, domain, k):
    """Yield every build that differs from base in exactly k changeable columns."""
    for cols in itertools.combinations(changeable, k):
        alts = [
            [v for v in domain[c] if str(v) != str(base[c])] for c in cols
        ]
        for combo in itertools.product(*alts):
            cand = dict(base)
            cand.update(zip(cols, combo))
            yield cand


def fix(train_csv, row, frozen=()) -> dict:
    """Return the row's inputs with as few values changed as possible so it is predicted good.

    `row` maps input column names to values; goal columns are absent or ignored.
    Columns named in `frozen` keep their original values. If the row is already
    predicted good, it is returned unchanged.
    """
    columns, rows = _load(train_csv)
    maximise, minimise, inputs = _split_columns(columns)
    goals = maximise + minimise

    if isinstance(frozen, str):
        frozen = (frozen,)
    frozen = set(frozen)

    input_set = set(inputs)
    base = {k: v for k, v in row.items() if k in input_set}
    if not goals or not rows:
        return dict(base)

    vecs = [v for v in (_vector(r, maximise, minimise) for r in rows) if v is not None]
    if not vecs:
        return dict(base)
    m = len(vecs[0])
    lo = [min(v[j] for v in vecs) for j in range(m)]
    hi = [max(v[j] for v in vecs) for j in range(m)]

    def nsum(vec):
        total = 0.0
        for j in range(m):
            if hi[j] > lo[j]:
                total += (vec[j] - lo[j]) / (hi[j] - lo[j])
        return total

    threshold = _quantile(sorted(nsum(v) for v in vecs), GOOD_QUANTILE)

    def score(cand):
        est = guess(train_csv, cand)
        vec = _vector({g: est[g][0] for g in goals}, maximise, minimise)
        return math.inf if vec is None else nsum(vec)

    base_score = score(base)
    if base_score <= threshold:
        return dict(base)

    changeable = [c for c in inputs if c in base and c not in frozen]
    if not changeable:
        return dict(base)

    domain = _domains(rows, changeable)
    best_cand, best_overall = dict(base), base_score
    evals = 0

    for k in range(1, len(changeable) + 1):
        best_k = None
        for cand in _changes(changeable, base, domain, k):
            if evals >= MAX_EVALS:
                break
            evals += 1
            s = score(cand)
            if best_k is None or s < best_k[0]:
                best_k = (s, cand)
            if s < best_overall:
                best_overall, best_cand = s, cand
        if best_k is not None and best_k[0] <= threshold:
            return dict(best_k[1])
        if evals >= MAX_EVALS:
            break

    return dict(best_cand)

