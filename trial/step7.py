"""Step 7: find a good flag set while measuring only a few builds.

choose() sees the flag settings of every build in blind_csv but none of their
goal values. It may measure at most `budget` builds by calling label(i). It
first measures builds spread across the flag space (farthest-point sampling),
then repeatedly measures the unmeasured build with the best predicted goal
score. The prediction is a distance-weighted average over the nearest measured
builds, minus a bonus for builds far from anything measured so far.

A goal score is the mean of the measured goals, each normalised over the
measured builds, with '+' goals negated so that lower is better. With
goal=..., only that goal counts, so the best choice for speed alone or CPU
alone can differ from the best overall choice. The measured build with the
best goal score is returned.
"""

import math

from step1 import _load, _split_columns

SEED_MIN = 3
NEIGHBOURS = 3
KAPPA = 0.5
EPS = 1e-6


def _as_finite(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _spec(rows, inputs):
    """Per input column: ('num', span) if every value is a float, else ('cat', 0.0)."""
    spec = {}
    for c in inputs:
        vals = [r.get(c) for r in rows]
        if vals and all(isinstance(v, float) for v in vals):
            finite = [v for v in vals if math.isfinite(v)]
            if finite:
                spec[c] = ("num", max(finite) - min(finite))
                continue
        spec[c] = ("cat", 0.0)
    return spec


def _make_distance(rows, inputs):
    """Return distance(a, b) between row indices: mean per-column difference in [0, 1]."""
    spec = _spec(rows, inputs)
    width = max(1, len(inputs))

    def distance(a, b):
        ra, rb = rows[a], rows[b]
        d = 0.0
        for c in inputs:
            kind, span = spec[c]
            x, y = ra.get(c), rb.get(c)
            if kind == "num":
                if (
                    isinstance(x, float)
                    and isinstance(y, float)
                    and math.isfinite(x)
                    and math.isfinite(y)
                ):
                    d += abs(x - y) / span if span > 0 else 0.0
                else:
                    d += 1.0
            else:
                d += 0.0 if str(x) == str(y) else 1.0
        return d / width

    return distance


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Return the index of the best row found while calling label() at most budget times.

    blind_csv has every goal cell set to '?'. label(i) returns the goals of
    row i as a dict. If goal names one goal, only that goal is optimised.
    """
    columns, rows = _load(blind_csv)
    maximise, minimise, inputs = _split_columns(columns)

    if goal is not None:
        maximise = [g for g in maximise if g == goal]
        minimise = [g for g in minimise if g == goal]
        if not (maximise or minimise):
            raise ValueError(f"goal {goal!r} is not a goal column of {blind_csv}")

    names = maximise + minimise
    sign = [-1.0] * len(maximise) + [1.0] * len(minimise)
    n = len(rows)
    if n == 0:
        raise ValueError(f"{blind_csv} has no rows")
    if not names:
        raise ValueError("no goal columns to optimise")
    if budget < 1:
        raise ValueError("budget must be at least 1")
    budget = min(int(budget), n)
    m = len(names)

    distance = _make_distance(rows, inputs)

    vecs = {}                          # measured index -> minimisation vector or None
    labelled = []                      # measured indices, in measurement order
    dist_to = [[] for _ in range(n)]   # dist_to[i][k] = distance from i to labelled[k]

    def to_vector(values):
        vec = []
        for g, s in zip(names, sign):
            v = _as_finite(values.get(g))
            if v is None:
                return None
            vec.append(s * v)
        return vec

    def measure(i):
        if len(labelled) >= budget:
            raise RuntimeError("choose(): label budget exhausted")
        vecs[i] = to_vector(label(i))
        labelled.append(i)
        for r in range(n):
            if r not in vecs:
                dist_to[r].append(distance(r, i))

    def farthest():
        cand = [i for i in range(n) if i not in vecs]
        return max(cand, key=lambda i: (min(dist_to[i]), -i))

    def scores():
        """Mean normalised goal value (lower is better) for each usable measured row."""
        usable = [j for j in labelled if vecs[j] is not None]
        if not usable:
            return {}
        lo = [min(vecs[j][k] for j in usable) for k in range(m)]
        hi = [max(vecs[j][k] for j in usable) for k in range(m)]
        out = {}
        for j in usable:
            total = 0.0
            for k in range(m):
                if hi[k] > lo[k]:
                    total += (vecs[j][k] - lo[k]) / (hi[k] - lo[k])
            out[j] = total / m
        return out

    # Phase 1: spread the first measurements across the flag space.
    measure(0)
    seeds = min(budget, max(SEED_MIN, budget // 3))
    while len(labelled) < seeds:
        measure(farthest())

    # Phase 2: measure the unmeasured build that looks best under the prediction.
    while len(labelled) < budget:
        cand = [i for i in range(n) if i not in vecs]
        if not cand:
            break
        s = scores()
        if not s:
            pick = farthest()
        else:
            pos = {j: k for k, j in enumerate(labelled)}
            known = [(pos[j], s[j]) for j in s]
            best_acq, pick = None, None
            for i in cand:
                ds = dist_to[i]
                near = sorted((ds[k], sc) for k, sc in known)[:NEIGHBOURS]
                wsum = sum(1.0 / (d + EPS) for d, _ in near)
                pred = sum(sc / (d + EPS) for d, sc in near) / wsum
                acq = pred - KAPPA * min(ds)
                if best_acq is None or acq < best_acq:
                    best_acq, pick = acq, i
        measure(pick)

    # Return the measured build with the best goal score.
    final = scores()
    if not final:
        return labelled[0]
    return min(final, key=lambda j: (final[j], j))

