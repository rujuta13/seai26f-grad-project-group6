"""Step 21: group customers into shopper types, and find the best customers.

families(train_csv) clusters the customers on their input columns. Input
columns are standardised if numeric and one-hot encoded if categorical.
Identifier-like columns (integer-valued and almost every value different) and
free-text columns with many levels are left out, so the clusters describe how
customers behave rather than their row number or name. A seeded k-means is
fitted on a sample and then applied to every row.

best(train_csv) returns the Pareto-front customers on all goal columns, ordered
by front and then by a normalised goal score, at most 5% of the rows. Customers
with a missing goal value cannot be judged and are not returned.
"""
import math
import random

from step1 import _load

_K = 5                    # number of shopper types (families)
_MAX_LEVELS = 30          # categorical columns with more levels are dropped
_ID_MIN_DISTINCT = 500    # integer columns this unique look like identifiers
_FIT_SAMPLE = 3000        # rows used to fit the k-means centres
_MAX_ITER = 30            # Lloyd iterations on the sample
_SEED = 1
_TOP_FRACTION = 0.05      # share of rows returned by best()


def _is_goal(c):
    return c.endswith("+") or c.endswith("-")


def _is_input(c):
    return not _is_goal(c) and not c.endswith("X")


def _present(v):
    """True if v is a finite number or a non-blank string."""
    if isinstance(v, float):
        return math.isfinite(v)
    if isinstance(v, str):
        return bool(v.strip())
    return False


def _text(v):
    """Comparable text for a categorical cell, or None if missing."""
    if isinstance(v, float):
        return f"{v:g}" if math.isfinite(v) else None
    if isinstance(v, str):
        s = v.strip()
        return s or None
    return None


def _num(v):
    """v as a finite float, or None."""
    return v if isinstance(v, float) and math.isfinite(v) else None


def _block(cells):
    """Feature columns for one input column; each is a list with one value per row.

    Returns [] if the column is dropped.
    """
    present = [v for v in cells if _present(v)]
    if not present:
        return []
    if all(isinstance(v, float) for v in present):
        distinct = len(set(present))
        if (distinct > _ID_MIN_DISTINCT
                and distinct >= 0.9 * len(present)
                and all(v.is_integer() for v in present)):
            return []  # identifier-like: row numbers and the like
        mu = sum(present) / len(present)
        sd = math.sqrt(sum((v - mu) ** 2 for v in present) / len(present))
        if sd == 0.0:
            return []
        return [[(v - mu) / sd if _present(v) else 0.0 for v in cells]]
    texts = [_text(v) for v in cells]
    levels = sorted({t for t in texts if t is not None})
    if not levels or len(levels) > _MAX_LEVELS:
        return []
    return [[1.0 if t == lv else 0.0 for t in texts] for lv in levels]


def _matrix(header, rows):
    """Feature matrix: one list of floats per row, over all usable input columns."""
    columns = []
    for c in header:
        if _is_input(c):
            columns.extend(_block([r.get(c) for r in rows]))
    return [[col[i] for col in columns] for i in range(len(rows))]


def _sq(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _fit(X, k, rng, max_iter):
    """Lloyd's k-means with k-means++ seeding. Returns the list of centres."""
    n = len(X)
    d = len(X[0])
    centers = [list(X[rng.randrange(n)])]
    while len(centers) < k:
        dist = [min(_sq(x, c) for c in centers) for x in X]
        total = sum(dist)
        if total <= 0.0:
            centers.append(list(X[rng.randrange(n)]))
            continue
        r = rng.random() * total
        acc = 0.0
        pick = n - 1
        for i, di in enumerate(dist):
            acc += di
            if acc >= r:
                pick = i
                break
        centers.append(list(X[pick]))

    labels = [-1] * n
    for _ in range(max_iter):
        changed = False
        for i, x in enumerate(X):
            best_c = min(range(k), key=lambda c: _sq(x, centers[c]))
            if labels[i] != best_c:
                labels[i] = best_c
                changed = True
        if not changed:
            break
        sums = [[0.0] * d for _ in range(k)]
        counts = [0] * k
        for x, lab in zip(X, labels):
            counts[lab] += 1
            s = sums[lab]
            for j, v in enumerate(x):
                s[j] += v
        for c in range(k):
            if counts[c]:
                centers[c] = [v / counts[c] for v in sums[c]]
    return centers


def families(train_csv) -> list[int]:
    """One family (shopper type) label per data row of train_csv, in file order.

    Clusters are computed on the usable input columns (see the module
    docstring). Labels are renumbered 0..k-1 in order of first appearance.
    """
    header, rows = _load(train_csv)
    n = len(rows)
    if n == 0:
        return []

    X = _matrix(header, rows)
    if not X or not X[0]:
        return [0] * n

    k = min(_K, n)
    rng = random.Random(_SEED)
    m = min(n, _FIT_SAMPLE)
    sample = [X[i] for i in sorted(rng.sample(range(n), m))]
    centers = _fit(sample, k, rng, _MAX_ITER)

    raw = [min(range(k), key=lambda c: _sq(x, centers[c])) for x in X]

    relabel = {}
    out = []
    for lab in raw:
        if lab not in relabel:
            relabel[lab] = len(relabel)
        out.append(relabel[lab])
    return out


def _dominates(a, b):
    """True if a is at least as good as b on every goal and strictly better on one."""
    strict = False
    for x, y in zip(a, b):
        if x < y:
            return False
        if x > y:
            strict = True
    return strict


def best(train_csv) -> list[int]:
    """Row indices of the unusually good customers (best on all goals together).

    Goals ending in '+' are maximised and goals ending in '-' are minimised.
    Rows come in Pareto-front order, and within each front by a normalised sum
    of goal scores. At most 5% of rows are returned.
    """
    header, rows = _load(train_csv)
    n = len(rows)
    cap = int(_TOP_FRACTION * n)
    if n == 0 or cap == 0:
        return []

    goals = [
        (c, 1.0 if c.endswith("+") else -1.0)
        for c in header if _is_goal(c)
    ]
    if not goals:
        return []

    # Oriented goal vectors (higher is better) for rows with every goal finite.
    V = {}
    for i, r in enumerate(rows):
        vec = []
        for c, sg in goals:
            f = _num(r.get(c))
            if f is None:
                break
            vec.append(sg * f)
        else:
            V[i] = tuple(vec)
    if not V:
        return []

    ids = list(V)
    g = len(goals)
    lo = [min(V[i][j] for i in ids) for j in range(g)]
    hi = [max(V[i][j] for i in ids) for j in range(g)]
    score = {}
    for i in ids:
        total = 0.0
        for j in range(g):
            span = hi[j] - lo[j]
            total += 1.0 if span == 0 else (V[i][j] - lo[j]) / span
        score[i] = total

    # Lexicographic descending order: a dominator always comes before what it
    # dominates, so each front can be built by checking only earlier front members.
    remaining = sorted(ids, key=lambda i: V[i], reverse=True)
    order = []
    while remaining and len(order) < cap:
        front = []
        for i in remaining:
            v = V[i]
            for j in front:
                if _dominates(V[j], v):
                    break
            else:
                front.append(i)
        front.sort(key=lambda i: (-score[i], i))
        order.extend(front)
        taken = set(front)
        remaining = [i for i in remaining if i not in taken]

    return order[:cap]

