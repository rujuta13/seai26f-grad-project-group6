"""Step 8: fix a slow build with as few flag changes as possible.

fix(train_csv, row, frozen) returns a copy of a bad build's input flags with
the fewest values changed that make it "good", using the rule from step6.why:
each goal is scaled over the training builds (0 worst, 1 best), the mean of
those scaled values is the build's score, and the build is good if at least
half of the training builds score no higher.

A candidate's goal values are unknown, so they are estimated from its nearest
training builds, as in step2.guess. Changes are searched by size: all single
flag changes, then all pairs, and so on up to _K_MAX, so the first good result
uses the fewest changes. If no small change set works within the evaluation
budget, a greedy search takes over. Frozen columns, ignored 'X' columns and
goal columns are never changed.

If no good build is found, fix() returns the highest-scoring build it tried
(or the original row if nothing scored higher), which may still be bad.
"""
import itertools
import math

from step1 import _load

_K_MAX = 6          # largest change set tried exhaustively
_MAX_EVALS = 1500   # cap on candidate evaluations (each scans the training data)
_MAX_OPTIONS = 8    # values tried per numeric column


def _is_goal(c):
    return c.endswith("+") or c.endswith("-")


def _is_input(c):
    return not _is_goal(c) and not c.endswith("X")


def _finite(v):
    """Return v as a finite float, or None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _level(v):
    """Comparable key for a cell: finite float, non-empty string, or None."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return None


def _median(vals):
    """Median of a non-empty sorted list (matches step2's 0.5 quantile)."""
    m = len(vals)
    mid = m // 2
    if m % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _apply(base, changes):
    """Copy of base with the (column, value) changes applied."""
    out = dict(base)
    for c, v in changes:
        out[c] = v
    return out


class _Model:
    """Training data prepared once, so many candidate rows can be scored."""

    def __init__(self, train_csv):
        header, rows = _load(train_csv)
        self.rows = rows
        self.n = len(rows)
        self.goals = []
        for c in header:
            if c.endswith("+"):
                self.goals.append((c, 1.0))
            elif c.endswith("-"):
                self.goals.append((c, -1.0))
        self.inputs = [c for c in header if _is_input(c)]
        self.goal_vals = {
            c: [_finite(r.get(c)) for r in rows] for c, _ in self.goals
        }

        complete = []
        for i in range(self.n):
            v = self._oriented_row(i)
            if v is not None:
                complete.append(v)
        self.scorable = bool(complete) and bool(self.goals)
        if self.scorable:
            g = len(self.goals)
            self.lo = [min(v[j] for v in complete) for j in range(g)]
            self.hi = [max(v[j] for v in complete) for j in range(g)]
            self.train_scores = [self.score_vec(v) for v in complete]

        self.options = {c: self._options(c) for c in self.inputs}

    def _oriented_row(self, i):
        """Goal values of training row i with signs applied, or None if missing."""
        vec = []
        for c, sg in self.goals:
            f = self.goal_vals[c][i]
            if f is None:
                return None
            vec.append(sg * f)
        return vec

    def _options(self, c):
        """Values to try for input column c, drawn from the training data."""
        levels = set()
        for r in self.rows:
            lv = _level(r.get(c))
            if lv is not None:
                levels.add(lv)
        vals = sorted(levels, key=lambda v: (isinstance(v, str), v))
        if len(vals) > _MAX_OPTIONS and all(isinstance(v, float) for v in vals):
            m = len(vals)
            idx = sorted({
                round(k * (m - 1) / (_MAX_OPTIONS - 1))
                for k in range(_MAX_OPTIONS)
            })
            vals = [vals[i] for i in idx]
        return vals

    def score_vec(self, vec):
        """Mean of the clamped 0..1 normalised goals (higher is better)."""
        total = 0.0
        for j, x in enumerate(vec):
            span = self.hi[j] - self.lo[j]
            t = 1.0 if span == 0 else (x - self.lo[j]) / span
            total += min(1.0, max(0.0, t))
        return total / len(vec)

    def _estimate(self, row):
        """Oriented goal estimate for a candidate row, from its nearest training builds."""
        cols = [c for c in self.inputs if c in row]
        dists = []
        for r in self.rows:
            d = 0.0
            for c in cols:
                a = row[c]
                b = r.get(c)
                if isinstance(a, float) and isinstance(b, float):
                    d += abs(a - b)
                elif a != b:
                    d += 1.0
            dists.append(d)
        order = sorted(range(self.n), key=lambda i: (dists[i], i))
        k = min(self.n, max(10, int(math.sqrt(self.n))))
        near = order[:k]
        vec = []
        for c, sg in self.goals:
            gv = self.goal_vals[c]
            vals = sorted(gv[i] for i in near if gv[i] is not None)
            if len(vals) < 3:
                vals = sorted(v for v in gv if v is not None)
            if not vals:
                return None
            vec.append(sg * _median(vals))
        return vec

    def evaluate(self, row):
        """(score, good) for a candidate row, or None if it cannot be estimated."""
        vec = self._estimate(row)
        if vec is None:
            return None
        s = self.score_vec(vec)
        at_most = sum(1 for t in self.train_scores if t <= s) / len(self.train_scores)
        return s, at_most >= 0.5


def fix(train_csv, row, frozen=()) -> dict:
    """Inputs of `row` with as few values changed as possible so it becomes good.

    `row` is a bad build's input columns (goals removed). Columns named in
    `frozen` are never changed. Returns a new dict with the same keys as `row`.
    The original row is returned if it is already good or no change helps.
    """
    if isinstance(frozen, str):
        frozen = (frozen,)
    base = dict(row)
    model = _Model(train_csv)
    if not model.scorable:
        return base
    frozen = set(frozen)

    base_res = model.evaluate(base)
    if base_res is not None and base_res[1]:
        return base

    start = base_res[0] if base_res is not None else -math.inf
    best_score = start
    best_row = base
    evals = 0

    def consider(cand, score):
        nonlocal best_score, best_row
        if score > best_score:
            best_score, best_row = score, cand

    cols = [c for c in model.inputs if c in base and c not in frozen]
    moves = []
    for c in cols:
        cur = _level(base.get(c))
        for v in model.options[c]:
            if v != cur:
                moves.append((c, v))
    if not moves:
        return base

    # Exhaustive search by number of changed flags: the first size with a
    # good build gives the fewest changes; ties go to the higher score.
    for k in range(1, min(_K_MAX, len(cols)) + 1):
        found = []
        for combo in itertools.combinations(moves, k):
            if len({c for c, _ in combo}) < k:
                continue
            if evals >= _MAX_EVALS:
                break
            cand = _apply(base, combo)
            res = model.evaluate(cand)
            evals += 1
            if res is None:
                continue
            consider(cand, res[0])
            if res[1]:
                found.append((res[0], cand))
        if found:
            return max(found, key=lambda t: t[0])[1]
        if evals >= _MAX_EVALS:
            break

    # Greedy fallback: keep taking the single change that helps most.
    changes = []
    used = set()
    cur = start
    while evals < _MAX_EVALS:
        step = None
        for c, v in moves:
            if c in used:
                continue
            if evals >= _MAX_EVALS:
                break
            cand = _apply(base, changes + [(c, v)])
            res = model.evaluate(cand)
            evals += 1
            if res is None:
                continue
            if step is None or res[0] > step[0]:
                step = (res[0], res[1], (c, v), cand)
        if step is None or step[0] <= cur:
            break
        cur, good, mv, cand = step
        changes.append(mv)
        used.add(mv[0])
        consider(cand, cur)
        if good:
            return cand

    return best_row

