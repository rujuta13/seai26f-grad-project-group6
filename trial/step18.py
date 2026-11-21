"""Step 18: fix. The smallest change to a player's inputs that makes him shortlist-good.

The client asks what a close player would need to improve to make the shortlist,
and what training focus would turn a weak penalty-taker into a reliable one. Both
come down to the same question: which input values to change, and how few.

Method
  - "Good" follows the step11 shortlist standard. A player's estimated goal score
    (each goal min-max scaled over the training players with complete goal values,
    0 = worst, 1 = best, averaged over goals) must reach the lowest score among the
    players step11.best returns.
  - Goal estimates come from step12.guess, so the search changes only inputs the
    model uses. Identifier-like numeric columns and many-level text columns are
    never changed.
  - Candidate values are real values seen in train_csv. For a numeric column these
    are the nearest training values just below and just above the player's current
    value, plus an even spread of the column's distinct values. For a text column
    they are its most common levels.
  - Search is a beam search over change sets, by size. Every single change is tried,
    then the BEAM best single changes are extended to pairs, and so on up to
    MAX_CHANGES. The first size that reaches the good threshold wins. Among those,
    the change set with the smallest total movement is returned. Text changes count
    1 each, numeric changes count their size as a share of the column's training range.
  - If nothing good is found within the limits, the change set with the highest
    estimated score is returned.
  - Columns named in `frozen` are never changed. A row already estimated to be good
    is returned unchanged.

The search runs on estimates, so a returned change set is a suggestion to test. The
beam search is not exhaustive, so it can miss a smaller fix.
"""
import bisect
import os
from collections import Counter

from step11 import _finite, _load, best
from step12 import _spec, _text, guess

MAX_CHANGES = 4        # largest number of inputs changed in one fix
BEAM = 8               # change sets kept per size when searching for larger fixes
MAX_SPREAD = 8         # evenly spread distinct training values offered per numeric column
MAX_LEVELS = 8         # most common text levels offered per text column
EVAL_BUDGET = 20000    # cap on candidate evaluations, to keep runtime bounded
EPS = 1e-12
_CACHE = {}


def _spread_values(vals, limit):
    """At most `limit` values from an ascending list, evenly spread from first to last."""
    if len(vals) <= limit:
        return list(vals)
    idx = sorted({round(i * (len(vals) - 1) / (limit - 1)) for i in range(limit)})
    return [vals[i] for i in idx]


def _vec_score(vec, lo, hi):
    """Step11-scale score of an oriented goal vector (higher is better), clamped to the
    training range, averaged over goals. A constant goal contributes 0."""
    total = 0.0
    for j, v in enumerate(vec):
        span = hi[j] - lo[j]
        if span > 0:
            total += (min(max(v, lo[j]), hi[j]) - lo[j]) / span
    return total / len(vec)


def _data(train_csv):
    """Cached training data: changeable columns, their candidate values, goal scale,
    and the good-build threshold (the lowest score among the step11 shortlist)."""
    stamp = (train_csv, os.path.getmtime(train_csv))
    hit = _CACHE.get(stamp)
    if hit is not None:
        return hit

    rows, inputs, goals = _load(train_csv)
    d = {"goals": goals, "kind": {}, "dv": {}, "spread": {}, "span": {},
         "levels": {}, "cols": [], "lo": [], "hi": [], "threshold": None}

    if rows and goals:
        nums, cats = _spec(rows, inputs)

        for c, _, _, _ in nums:
            vals = sorted({x for x in (_finite(r.get(c)) for r in rows) if x is not None})
            if len(vals) < 2:
                continue
            d["kind"][c] = "num"
            d["dv"][c] = vals
            d["span"][c] = vals[-1] - vals[0]
            d["spread"][c] = _spread_values(vals, MAX_SPREAD)
            d["cols"].append(c)

        for c, levels in cats:
            counts = Counter(_text(r.get(c)) for r in rows)
            ranked = sorted((lv for lv in levels if lv), key=lambda lv: (-counts[lv], lv))
            d["kind"][c] = "cat"
            d["levels"][c] = ranked[:MAX_LEVELS]
            d["cols"].append(c)

        usable = {}
        for i, r in enumerate(rows):
            vec = []
            for c, sign in goals:
                x = _finite(r.get(c))
                if x is None:
                    break
                vec.append(sign * x)
            else:
                usable[i] = vec

        if usable:
            g = len(goals)
            lo = [min(v[j] for v in usable.values()) for j in range(g)]
            hi = [max(v[j] for v in usable.values()) for j in range(g)]
            d["lo"], d["hi"] = lo, hi
            good = [i for i in best(train_csv) if i in usable]
            if good:
                d["threshold"] = min(_vec_score(usable[i], lo, hi) for i in good)

    _CACHE.clear()
    _CACHE[stamp] = d
    return d


def _alternatives(c, cur, d):
    """(value, movement) pairs for moving column c away from its current cell.
    movement is the change as a share of the column's training range (1 for text
    columns and for a missing current cell)."""
    if d["kind"][c] == "num":
        dv = d["dv"][c]
        span = d["span"][c]
        x = _finite(cur)
        cands = set(d["spread"][c])
        if x is not None:
            i = bisect.bisect_left(dv, x)
            if i > 0:
                cands.add(dv[i - 1])          # nearest training value just below
            j = bisect.bisect_right(dv, x)
            if j < len(dv):
                cands.add(dv[j])              # nearest training value just above
            cands.discard(x)
        out = []
        for v in sorted(cands):
            mv = 1.0 if x is None else abs(v - x) / span
            out.append((v, mv))
        return out

    cur_t = _text(cur)
    return [(lv, 1.0) for lv in d["levels"][c] if lv != cur_t]


def fix(train_csv, row, frozen=()) -> dict:
    """Return the row's inputs with as few values changed as possible so it becomes good.

    `row` is a player with goals removed. Columns named in `frozen` are never changed.
    If the row is already estimated to be good, it is returned unchanged. If no good
    version is found within the search limits, the change set with the highest
    estimated score is returned.
    """
    d = _data(train_csv)
    out = dict(row)
    threshold = d["threshold"]
    if threshold is None or not d["cols"]:
        return out

    frozen = set(frozen)

    def score(cand):
        est = guess(train_csv, cand)
        vec = [sign * est[name][0] for name, sign in d["goals"]]
        return _vec_score(vec, d["lo"], d["hi"])

    base = score(out)
    if base >= threshold - EPS:
        return out

    opts = {}
    for c in d["cols"]:
        if c in frozen:
            continue
        alts = _alternatives(c, out.get(c), d)
        if alts:
            opts[c] = alts
    cols = list(opts)

    evals = 0
    seen = set()
    best_any = (base, ())
    good = []                      # (total movement, -score, changes) at the first good size
    beam = [(base, ())]
    for _depth in range(1, MAX_CHANGES + 1):
        pool = []
        for _, changes in beam:
            used = {c for c, _, _ in changes}
            for c in cols:
                if c in used:
                    continue
                for v, mv in opts[c]:
                    if evals >= EVAL_BUDGET:
                        break
                    new = changes + ((c, v, mv),)
                    key = frozenset((cc, vv) for cc, vv, _ in new)
                    if key in seen:
                        continue
                    seen.add(key)
                    cand = dict(out)
                    for cc, vv, _ in new:
                        cand[cc] = vv
                    s = score(cand)
                    evals += 1
                    pool.append((s, new))
                    if s > best_any[0]:
                        best_any = (s, new)
                    if s >= threshold - EPS:
                        good.append((sum(m for _, _, m in new), -s, new))
        if good or not pool or evals >= EVAL_BUDGET:
            break
        pool.sort(key=lambda t: -t[0])
        beam = pool[:BEAM]

    if good:
        chosen = min(good, key=lambda t: (t[0], t[1]))[2]
    else:
        chosen = best_any[1]

    for c, v, _ in chosen:
        out[c] = v
    return out

