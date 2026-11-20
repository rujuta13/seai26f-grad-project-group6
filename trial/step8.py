"""step8.py: fix. Find the fewest input values to change so a bad build becomes good.

fix() does not benchmark anything. Each candidate flag set gets its goals guessed
with the nearest-neighbour rule of step2.guess, is rated with the same
mean-normalised score and top-quarter cutoff as step6.why, and a search over flag
changes, fewest first, looks for a setting rated good. Frozen columns never change.

Reading, column splitting, missing-value filling, input encoding, goal scores,
the bad-quarter cut, the neighbour count and the goal vector come from steps 1,
2, 4, 5 and 6.
"""
from __future__ import annotations

import heapq
import math
from collections import Counter

from step1 import _columns, _dist2, _num, _read
from step2 import K_NEIGHBOURS, _fill, _quantile
from step4 import _encode, _fit, _objectives
from step5 import BAD_QUANTILE, _scores

MAX_CHANGES = 4         # most input values the search will change
BEAM_WIDTH = 8          # partial fixes kept from each level of the search
MAX_VALUES = 6          # most common training values tried for each input


def _candidates(rows: list[dict], c: str) -> list:
    """The most common usable training values of input column c."""
    vals = []
    for r in rows:
        v = r.get(c)
        if isinstance(v, float) and math.isfinite(v):
            vals.append(v)
        elif isinstance(v, str) and v != "":
            vals.append(v)
    return [v for v, _ in Counter(vals).most_common(MAX_VALUES)]


class _Rater:
    """Guesses goals for flag sets and rates them, as step2.guess and step6.why do."""

    def __init__(self, rows: list[dict], inputs: list[str],
                 gmax: list[str], gmin: list[str]):
        self.rows = rows
        self.inputs = inputs
        self.goals = gmax + gmin
        self.signs = [1.0] * len(gmax) + [-1.0] * len(gmin)
        self.spec = _fit(rows, inputs)
        self.feats = [_encode(r, inputs, self.spec) for r in rows]
        self.cands = {
            g: [i for i, r in enumerate(rows) if _num(r.get(g)) is not None]
            for g in self.goals
        }

        # Goal bounds and the top-quarter cutoff, exactly as step6.why builds them.
        obj = [_objectives(r, gmax, gmin) for r in rows]
        m = len(self.goals)
        self.lo: list[float] = []
        self.span: list[float] = []
        for j in range(m):
            fin = [o[j] for o in obj if math.isfinite(o[j])]
            a = min(fin) if fin else 0.0
            b = max(fin) if fin else 0.0
            self.lo.append(a)
            self.span.append(b - a)
        self.cutoff = _quantile(_scores(obj, m), 1.0 - BAD_QUANTILE)
        self.cache: dict[tuple, float] = {}

    def score(self, flags: dict) -> float:
        """Mean normalised score of the guessed goals for one flag set."""
        key = tuple(flags[c] for c in self.inputs)
        hit = self.cache.get(key)
        if hit is not None:
            return hit

        x = _encode(flags, self.inputs, self.spec)
        d2 = [_dist2(f, x) for f in self.feats]
        total = 0.0
        for j, (g, sign) in enumerate(zip(self.goals, self.signs)):
            near = heapq.nsmallest(K_NEIGHBOURS, self.cands[g], key=d2.__getitem__)
            if not near:
                continue  # no usable guess scores 0, as in step6.why
            est = sum(self.rows[i][g] for i in near) / len(near)
            v = sign * est
            if self.span[j] > 0:
                total += min(1.0, max(0.0, (v - self.lo[j]) / self.span[j]))
            else:
                total += 1.0
        s = total / len(self.goals)
        self.cache[key] = s
        return s

    def good(self, flags: dict) -> bool:
        return self.score(flags) >= self.cutoff


def fix(train_csv, row, frozen=()) -> dict:
    """Input values for the bad build `row` with as few values changed as possible
    so that it is rated good. Returns a dict over every input column.

    row: a build's input cells (goal cells are ignored). Missing cells are filled
         with training defaults, as in step2.guess.
    frozen: input column names that must not change. A single name or an iterable
            of names is accepted. Names that are not input columns raise ValueError.

    The search is a beam search over single-value changes, fewest changes first.
    Goals are guessed from the nearest training builds, so a returned setting is
    good by prediction, not by benchmark. If the row is already good, it is
    returned unchanged. If no setting rated good is found within MAX_CHANGES
    changes, the highest-rated setting found is returned instead.
    """
    header, rows = _read(str(train_csv))
    inputs, gmax, gmin = _columns(header)

    if isinstance(frozen, str):
        frozen = (frozen,)
    frozen = set(frozen)
    unknown = sorted(frozen - set(inputs))
    if unknown:
        raise ValueError(f"frozen names are not input columns: {unknown}")

    base = _fill(row, rows, inputs)
    if not rows or not inputs or not (gmax or gmin):
        return base

    rater = _Rater(rows, inputs, gmax, gmin)
    if rater.good(base):
        return base

    free = [c for c in inputs if c not in frozen]
    cands = {c: _candidates(rows, c) for c in free}
    seen = {tuple(base[c] for c in inputs)}
    beam: list[dict] = [base]
    best_flags, best_score = base, rater.score(base)

    for _ in range(MAX_CHANGES):
        children: list[tuple[float, dict]] = []
        good: list[tuple[float, dict]] = []
        for flags in beam:
            for c in free:
                if flags[c] != base[c]:
                    continue  # each column changes at most once
                for v in cands[c]:
                    if v == base[c]:
                        continue
                    child = dict(flags)
                    child[c] = v
                    key = tuple(child[k] for k in inputs)
                    if key in seen:
                        continue
                    seen.add(key)
                    s = rater.score(child)
                    children.append((s, child))
                    if s > best_score:
                        best_score, best_flags = s, child
                    if s >= rater.cutoff:
                        good.append((s, child))

        if good:
            # Fewest changes reached so far; among those, the highest rated.
            return dict(max(good, key=lambda t: t[0])[1])

        children.sort(key=lambda t: -t[0])
        beam = [child for _, child in children[:BEAM_WIDTH]]
        if not beam:
            break

    return dict(best_flags)

