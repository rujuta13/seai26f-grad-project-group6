"""Step 14: flow. Two sequential checks on scouting reports that arrive one at a time.

stream(train_csv, rows) keeps a report when it is worth following up: no build
seen so far (every training build and every earlier report) is at least as good
on every goal. Goals ending in '+' are maximised and goals ending in '-' are
minimised. A report with a missing or non-numeric goal value cannot be judged
and is not kept.

drift(train_csv, rows) warns at the first report after which the new reports
no longer look like the training data. After the m-th report, the first m values
of each tested column are compared with that column's training values by a
two-sample Kolmogorov-Smirnov test at level 0.05 / (m (m+1) g), where g is the
number of tested columns. Over all m these levels sum to 0.05, and the union
bound over columns keeps the chance of any false warning at most 0.05, however
long the stream runs. Tested columns are the numeric inputs (identifier-like
integers such as row numbers are skipped) and the numeric goals. Text columns
are not tested.
"""
import bisect
import math

from step1 import _load

_ALPHA = 0.05
_ID_MIN = 20   # integer-valued columns this long with almost all values distinct look like identifiers


def _is_goal(c):
    return c.endswith("+") or c.endswith("-")


def _finite(v):
    """Return v as a finite float, or None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _goals(header):
    """Return [(column, sign)]: +1.0 for '+' goals, -1.0 for '-' goals."""
    out = []
    for c in header:
        if c.endswith("+"):
            out.append((c, 1.0))
        elif c.endswith("-"):
            out.append((c, -1.0))
    return out


def _oriented(row, goals):
    """Goal values with signs applied (higher is always better), or None if any is missing."""
    vec = []
    for c, sg in goals:
        f = _finite(row.get(c))
        if f is None:
            return None
        vec.append(sg * f)
    return tuple(vec)


def _at_least_as_good(a, b):
    """True if a is at least as good as b on every goal."""
    return all(x >= y for x, y in zip(a, b))


def _admit(archive, v):
    """Try to add v to the archive of unbeaten builds.

    Returns False and changes nothing if some archived build is at least as
    good as v on every goal. Otherwise v is added, archived builds that v is
    at least as good as are dropped (anything they cover, v covers too), and
    True is returned.
    """
    for a in archive:
        if _at_least_as_good(a, v):
            return False
    archive[:] = [a for a in archive if not _at_least_as_good(v, a)]
    archive.append(v)
    return True


def stream(train_csv, rows) -> list[bool]:
    """Decide keep (True) or not (False) for each row, in arrival order.

    A row is kept when no training build and no earlier row is at least as
    good on every goal, so it extends the unbeaten front seen so far. Rows
    missing a goal value are not kept.
    """
    header, train = _load(train_csv)
    goals = _goals(header)

    archive = []
    if goals:
        for r in train:
            v = _oriented(r, goals)
            if v is not None:
                _admit(archive, v)

    out = []
    for row in rows:
        v = _oriented(row, goals) if goals else None
        out.append(False if v is None else _admit(archive, v))
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


def _looks_like_id(vals):
    """True for integer-valued columns with almost all values distinct (row numbers, ranks)."""
    return (
        len(vals) >= _ID_MIN
        and all(v.is_integer() for v in vals)
        and len(set(vals)) >= 0.9 * len(vals)
    )


def _tested(header, train):
    """Return [(column, sorted training values)] for the numeric columns to monitor.

    A column qualifies when at least 90% of its non-blank training cells are
    finite numbers. Identifier-like input columns are skipped; goal columns
    are always kept.
    """
    out = []
    for c in header:
        if c.endswith("X"):
            continue
        present = [
            v for v in (r.get(c) for r in train)
            if v is not None and not (isinstance(v, str) and not v.strip())
        ]
        if not present:
            continue
        vals = [v for v in present if isinstance(v, float) and math.isfinite(v)]
        if not vals or len(vals) < 0.9 * len(present):
            continue
        if not _is_goal(c) and _looks_like_id(vals):
            continue
        out.append((c, sorted(vals)))
    return out


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first row at which new rows stop looking like train_csv.

    After the m-th row, the first m finite values of each tested column are
    tested against that column's training values with a KS test at level
    0.05 / (m (m+1) g), where g is the number of tested columns. The first
    rejection warns at row index m-1. Returns None if no warning is raised.
    """
    header, train = _load(train_csv)
    rows = list(rows)
    if not rows:
        return None

    cols = _tested(header, train)
    if not cols:
        return None
    g = len(cols)

    seen = {c: [] for c, _ in cols}
    for m, row in enumerate(rows, start=1):
        for c, _ in cols:
            f = _finite(row.get(c))
            if f is not None:
                bisect.insort(seen[c], f)

        alpha = _ALPHA / (m * (m + 1) * g)
        for c, ref in cols:
            sample = seen[c]
            if not sample:
                continue
            d = _ks_stat(sample, ref)
            if _ks_pvalue(d, len(sample), len(ref)) < alpha:
                return m - 1
    return None

