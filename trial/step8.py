"""Step 8: fix. The smallest change to a bad build's flags that makes it good.

Method: candidate flag sets are scored with a nearest-neighbour estimate of their
goals (the same idea as step2.guess, on cached training data), converted to the
step1 goal score. A build counts as good when its estimated score reaches the
lowest score among the good builds that step1.best returns. Change sets are tried
in order of size (one setting, then two, ...), so the first hit is a fewest-flags
fix. Frozen columns are never changed. If nothing good is found within the search
limits, the change set with the highest estimated score is returned instead.
"""
import heapq
import itertools
import os

from step1 import _load, best

NEIGHBOURS = 15        # nearest training rows that inform each estimate
MAX_CHANGES = 4        # largest number of settings changed in one fix
MAX_ALTERNATIVES = 6   # values tried per column (evenly spread for many-valued columns)
EVAL_BUDGET = 2000     # cap on candidate evaluations, to keep runtime bounded
EPS = 1e-12
_CACHE = {}


def _text(v):
    if isinstance(v, str):
        return v
    return "" if v is None else str(v)


def _spread(vals, limit):
    """At most `limit` values from an ascending list, evenly spread from first to last."""
    if len(vals) <= limit:
        return list(vals)
    idx = sorted({round(i * (len(vals) - 1) / (limit - 1)) for i in range(limit)})
    return [vals[i] for i in idx]


def _oriented(row, goals):
    """Goal vector with higher always better, or None if any goal cell is unusable."""
    out = []
    for c, sign in goals:
        try:
            out.append(sign * float(row[c]))
        except (KeyError, TypeError, ValueError):
            return None
    return out


def _score(o, lo, hi):
    """Mean of the per-goal scaled values; a constant goal contributes 0, as in step1."""
    total = 0.0
    for j in range(len(o)):
        span = hi[j] - lo[j]
        if span > 0:
            total += (o[j] - lo[j]) / span
    return total / len(o)


def _block(c, v, d):
    """Encoded features of one input cell, mirroring step1's encoding."""
    if d["kinds"][c] == "num":
        try:
            return [float(v)]
        except (TypeError, ValueError):
            return [d["med"][c]]
    s = _text(v)
    return [1.0 if s == cat else 0.0 for cat in d["cats"][c]]


def _vector(r, d):
    return [x for c in d["inputs"] for x in _block(c, r.get(c), d)]


def _weighted_median(pairs):
    """pairs: sorted list of (value, weight) with positive weights."""
    total = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return v
    return pairs[-1][0]


def _data(train_csv):
    """Cached training data: encodings, goal scales, and the good-build threshold."""
    stamp = (train_csv, os.path.getmtime(train_csv))
    hit = _CACHE.get(stamp)
    if hit is not None:
        return hit

    rows, inputs, goals = _load(train_csv)
    kinds, cats, med, alt_vals = {}, {}, {}, {}
    for c in inputs:
        raw = [r.get(c) for r in rows]
        try:
            nums = sorted(float(v) for v in raw)
        except (TypeError, ValueError):
            kinds[c] = "cat"
            cats[c] = sorted({_text(v) for v in raw})
            alt_vals[c] = _spread(cats[c], MAX_ALTERNATIVES)
        else:
            kinds[c] = "num"
            med[c] = nums[len(nums) // 2] if nums else 0.0
            alt_vals[c] = _spread(sorted(set(nums)), MAX_ALTERNATIVES)

    d = {"inputs": inputs, "kinds": kinds, "cats": cats, "med": med,
         "threshold": None, "X": [], "Y": [], "lo": [], "hi": []}
    d["alts"] = {c: [(v, _block(c, v, d)) for v in alt_vals[c]] for c in inputs}

    if goals and rows:
        oriented = [_oriented(r, goals) for r in rows]
        pool = [i for i, o in enumerate(oriented) if o is not None]
        if pool:
            g = len(goals)
            lo = [min(oriented[i][j] for i in pool) for j in range(g)]
            hi = [max(oriented[i][j] for i in pool) for j in range(g)]
            d["lo"], d["hi"] = lo, hi
            d["X"] = [_vector(rows[i], d) for i in pool]
            d["Y"] = [oriented[i] for i in pool]
            good = [i for i in best(train_csv) if oriented[i] is not None]
            if good:
                d["threshold"] = min(_score(oriented[i], lo, hi) for i in good)

    _CACHE.clear()
    _CACHE[stamp] = d
    return d


def _predict(q, d):
    """Estimated goal score of a feature vector q, from its nearest training rows."""
    X, Y, lo, hi = d["X"], d["Y"], d["lo"], d["hi"]
    dist = [sum((a - b) * (a - b) for a, b in zip(x, q)) for x in X]
    nb = heapq.nsmallest(min(NEIGHBOURS, len(X)), range(len(X)), key=dist.__getitem__)
    g = len(lo)
    total = 0.0
    for j in range(g):
        pairs = sorted((Y[i][j], 1.0 / (1.0 + dist[i])) for i in nb)
        est = _weighted_median(pairs)
        span = hi[j] - lo[j]
        if span > 0:
            total += (est - lo[j]) / span
    return total / g


def _candidates(cols, opts):
    """Change sets in order of size: each is a tuple of (column, value, block)."""
    for k in range(1, min(MAX_CHANGES, len(cols)) + 1):
        for combo in itertools.combinations(cols, k):
            for picks in itertools.product(*(opts[c] for c in combo)):
                yield tuple((c, v, blk) for c, (v, blk) in zip(combo, picks))


def fix(train_csv, row, frozen=()) -> dict:
    """Return the row's inputs with as few values changed as possible so it becomes good.

    Columns named in `frozen` are never changed. If the row is already estimated to
    be good, it is returned unchanged. If no good version is found within the search
    limits, the change set with the highest estimated score is returned.
    """
    d = _data(train_csv)
    out = dict(row)
    inputs = d["inputs"]
    threshold = d["threshold"]
    if threshold is None or not inputs:
        return out

    frozen = set(frozen)
    base = {c: _block(c, row.get(c), d) for c in inputs}
    offs, q0, pos = {}, [], 0
    for c in inputs:
        offs[c] = (pos, pos + len(base[c]))
        q0.extend(base[c])
        pos += len(base[c])

    opts = {}
    for c in inputs:
        if c in frozen:
            continue
        alt = [(v, blk) for v, blk in d["alts"][c] if blk != base[c]]
        if alt:
            opts[c] = alt
    cols = [c for c in inputs if c in opts]

    base_score = _predict(q0, d)
    if base_score >= threshold - EPS:
        return out

    evals = 0
    best_good = None           # (score, changes) for the smallest change size that works
    best_any = (base_score, ())
    for changes in _candidates(cols, opts):
        k = len(changes)
        if best_good is not None and k > len(best_good[1]):
            break
        if evals >= EVAL_BUDGET:
            break
        evals += 1

        q = list(q0)
        for c, _, blk in changes:
            s, e = offs[c]
            q[s:e] = blk
        score = _predict(q, d)

        if score >= threshold - EPS:
            if best_good is None or score > best_good[0]:
                best_good = (score, changes)
        if score > best_any[0]:
            best_any = (score, changes)

    chosen = best_good[1] if best_good is not None else best_any[1]
    for c, v, _ in chosen:
        out[c] = v
    return out

