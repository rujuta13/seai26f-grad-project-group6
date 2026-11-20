"""Step 4: streaming triage of nightly benchmark rows.

stream() keeps a row when no known row (training data or earlier stream rows)
beats it on every goal at once, that is, when it lies on the Pareto front of
everything seen so far. Only goal columns are used.

drift() compares the goal values of incoming rows with the training goal
values using a two-sample Kolmogorov-Smirnov test, re-run after every new row.
The m-th row is tested at level alpha_m = 0.05 / (m (m+1)); these levels sum
to 0.05 over all m, and each goal's share is divided by the number of goals.
By the union bound, the chance of any false warning over the whole stream is
at most 0.05, however long the stream runs.
"""
import bisect
import math

from step1 import _load

_ALPHA = 0.05


def _goals(header):
    """Return [(column, sign)]: sign +1.0 for '+' goals, -1.0 for '-' goals."""
    out = []
    for c in header:
        if c.endswith("+"):
            out.append((c, 1.0))
        elif c.endswith("-"):
            out.append((c, -1.0))
    return out


def _finite(v):
    """Return v as a finite float, or None if it is not a finite number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _oriented(row, goals):
    """Goal values with signs applied (higher is always better), or None."""
    vec = []
    for c, sg in goals:
        f = _finite(row.get(c))
        if f is None:
            return None
        vec.append(sg * f)
    return vec


def _dominates(a, b):
    """True if a is at least as good as b on every goal and better on one."""
    strict = False
    for x, y in zip(a, b):
        if x < y:
            return False
        if x > y:
            strict = True
    return strict


def _insert(archive, v):
    """Try to add v to a Pareto archive.

    Returns True if no archive member dominates v (v is then added, and any
    members v dominates are dropped). Returns False and changes nothing
    otherwise. Dropping dominated members is safe: whatever they dominate,
    v also dominates, by transitivity.
    """
    for a in archive:
        if _dominates(a, v):
            return False
    archive[:] = [a for a in archive if not _dominates(v, a)]
    archive.append(v)
    return True


def stream(train_csv, rows) -> list[bool]:
    """Decide keep (True) or not (False) for each row, in arrival order.

    A row is kept when no training row and no earlier kept or non-kept row
    beats it on all goals at once; that is, it extends the Pareto front of
    everything seen so far. Rows with missing or non-numeric goal values are
    not kept.
    """
    header, train = _load(train_csv)
    goals = _goals(header)

    archive = []
    if goals:
        for r in train:
            v = _oriented(r, goals)
            if v is not None:
                _insert(archive, v)

    out = []
    for row in rows:
        if not goals:
            out.append(False)
            continue
        v = _oriented(row, goals)
        if v is None:
            out.append(False)
            continue
        out.append(_insert(archive, v))
    return out


def _ks_stat(a, b):
    """Two-sample Kolmogorov-Smirnov statistic for sorted non-empty lists."""
    n1, n2 = len(a), len(b)
    i = j = 0
    d = 0.0
    while i < n1 and j < n2:
        x = a[i] if a[i] <= b[j] else b[j]
        while i < n1 and a[i] == x:
            i += 1
        while j < n2 and b[j] == x:
            j += 1
        d = max(d, abs(i / n1 - j / n2))
    return d


def _ks_pvalue(d, n1, n2):
    """Asymptotic two-sided p-value for KS statistic d with sizes n1, n2."""
    en = math.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 0.2:
        return 1.0
    total = 0.0
    for k in range(1, 101):
        term = math.exp(-2.0 * k * k * lam * lam)
        total += (-1.0) ** (k - 1) * term
        if term < 1e-12:
            break
    return min(1.0, max(0.0, 2.0 * total))


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first row at which new results stop matching training.

    After the m-th incoming row, the first m values of each goal are tested
    against the training values of that goal with a KS test at level
    0.05 / (m (m+1) g), where g is the number of goals tested. The first
    rejection warns at row index m-1. Returns None if no warning is raised.
    """
    header, train = _load(train_csv)
    goals = _goals(header)
    rows = list(rows)
    if not goals or not rows:
        return None

    ref = {}
    for c, _ in goals:
        vals = []
        for r in train:
            f = _finite(r.get(c))
            if f is not None:
                vals.append(f)
        vals.sort()
        ref[c] = vals

    testable = [c for c, _ in goals if ref[c]]
    if not testable:
        return None
    g = len(testable)

    new = {c: [] for c in testable}
    for m, row in enumerate(rows, start=1):
        for c in testable:
            f = _finite(row.get(c))
            if f is not None:
                bisect.insort(new[c], f)

        alpha = _ALPHA / (m * (m + 1)) / g
        for c in testable:
            if not new[c]:
                continue
            d = _ks_stat(new[c], ref[c])
            if _ks_pvalue(d, len(new[c]), len(ref[c])) < alpha:
                return m - 1
    return None

