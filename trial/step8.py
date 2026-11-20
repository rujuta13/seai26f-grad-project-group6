"""Step 8: fix. The fewest input changes that make a bad flag set good.

Method: a bounded search over input changes. Candidate change sets are tried
with 1 changed column, then 2, and so on (up to _MAX_CHANGES), so the first
level that yields a good row has the fewest changes. Each candidate's goals are
predicted with step2.guess and rated the way step6.why rates a row: per goal,
the share of training builds it is at least as good as (direction aware),
averaged over goals. A row is good when that average exceeds 0.5.

Columns named in `frozen` are never changed. Replacement values for a column
are the most common values seen in training (at most _MAX_ALTS per column).
If no good row is found within the budget, the best-rated change set found is
returned.
"""

import bisect
import itertools
import math
from collections import Counter
from functools import lru_cache

import step1
import step2

_MAX_ALTS = 6       # replacement values per column (most common in training first)
_MAX_CHANGES = 4    # largest number of columns the search may change
_MAX_EVALS = 3000   # cap on predictions; each one scans the training set
_GOOD = 0.5         # average share above which a row counts as good (as in step6.why)


def _is_num(v):
    return isinstance(v, float) and not math.isnan(v)


def _usable(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


@lru_cache(maxsize=4)
def _model(train_csv):
    """Parse train_csv once. Returns (inputs, signed goals, pools, alternatives).

    signed goals: [(goal, sign)], sign +1 for maximised and -1 for minimised.
    pools: {goal: sorted sign-adjusted training values}, so higher is better.
    alternatives: {input: most common training values, most common first}.
    """
    header, rows = step1._read(train_csv)
    inputs, goals_max, goals_min = step1._classify(header)
    signed = [(g, 1.0) for g in goals_max] + [(g, -1.0) for g in goals_min]
    pools = {
        g: sorted(sign * r[g] for r in rows if _is_num(r.get(g)))
        for g, sign in signed
    }
    alternatives = {}
    for name in inputs:
        counts = Counter(v for v in (r.get(name) for r in rows) if _usable(v))
        alternatives[name] = [v for v, _ in counts.most_common(_MAX_ALTS + 1)]
    return inputs, signed, pools, alternatives


def _rate(train_csv, cand, signed, pools):
    """Average share of training builds the predicted goals are at least as good as."""
    est = step2.guess(train_csv, cand)
    shares = []
    for g, sign in signed:
        pool = pools[g]
        if not pool:
            continue
        e = est.get(g, (math.nan,))[0]
        if not _is_num(e):
            shares.append(0.0)  # no usable prediction counts as worst
        else:
            shares.append(bisect.bisect_right(pool, sign * e) / len(pool))
    return sum(shares) / len(shares) if shares else 0.0


def _moves(base, mutable, alternatives):
    """Yield (number of changed columns, candidate row), fewest changes first."""
    for k in range(1, min(_MAX_CHANGES, len(mutable)) + 1):
        for cols in itertools.combinations(mutable, k):
            choices = [
                [v for v in alternatives[c] if v != base.get(c)][:_MAX_ALTS]
                for c in cols
            ]
            for values in itertools.product(*choices):
                cand = dict(base)
                cand.update(zip(cols, values))
                yield k, cand


def fix(train_csv, row, frozen=()) -> dict:
    """Same inputs as `row`, with as few values changed as possible so it becomes good.

    row: a bad row with goal columns removed.
    frozen: input column names that must not change (a single name is also accepted).
    Returns {input: value} for the input columns of `row`. A row that is already
    good is returned unchanged. If no good row is found within the search budget,
    the best-rated change set found is returned.
    """
    inputs, signed, pools, alternatives = _model(train_csv)
    if isinstance(frozen, str):
        frozen = (frozen,)
    frozen = set(frozen)

    base = {n: row[n] for n in inputs if n in row}
    mutable = [n for n in inputs if n not in frozen]

    def finish(cand):
        return {n: cand[n] for n in inputs if n in cand}

    if not any(pools.values()):
        return finish(base)  # nothing to rate against

    base_score = _rate(train_csv, base, signed, pools)
    if base_score > _GOOD:
        return finish(base)

    best_score, best = base_score, base
    level, level_best, evals = 0, None, 0
    for k, cand in _moves(base, mutable, alternatives):
        if k != level:
            # Every change set with `level` changes has been tried; stop if one was good.
            if level_best is not None and level_best[0] > _GOOD:
                return finish(level_best[1])
            level, level_best = k, None
        if evals >= _MAX_EVALS:
            break
        s = _rate(train_csv, cand, signed, pools)
        evals += 1
        if level_best is None or s > level_best[0]:
            level_best = (s, cand)
        if s > best_score:
            best_score, best = s, cand

    if level_best is not None and level_best[0] > _GOOD:
        return finish(level_best[1])
    return finish(best)

