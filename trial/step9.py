"""Step 9: spot. Flag benchmark results that look like glitches or unlike anything seen.

odd(train_csv, rows) gives each row a suspicion score (higher means more suspicious).
Three signals are combined:

  1. Unfamiliar flag set: distance to the nearest training flag set, using step2's
     mismatch measure. A flag set unlike any training row is "unlike anything seen".
  2. Odd goal value: per goal, the robust z-score of the value against the training
     values (median and MAD, falling back to standard deviation). A missing,
     non-numeric or infinite goal cell is a glitch or typo and scores as suspicious.
  3. Disagreement with exact twins: when a training row has exactly the same flag
     set, the goal value is compared with what those twins measured. A value far
     from its twins looks like a glitch.

Each row's score is its largest signal plus a small share of the average signal,
so one strong signal flags a row and weaker signals still rank it above a clean row.
"""

import math
import statistics

import step2

_Z_LO = 3.0                 # robust z-score below which a goal value is not odd
_Z_SPAN = 7.0               # z-score range over which suspicion rises from 0 to 1
_MISSING = 1.0              # suspicion for a non-numeric or non-finite goal cell
_TWIN_DIST = 0.0            # flag-set mismatch counted as an exact twin
_TWINS = 7                  # most twins used for the local comparison
_NOVEL_SCALE = step2._FAR   # flag-set mismatch at which a row is fully unfamiliar
_MIX = 0.1                  # weight of the average signal added to the largest one


def _finite(v):
    return isinstance(v, float) and math.isfinite(v)


def _median(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def _goal_stats(goal_values):
    """Per goal: (median, scale) of the training values. Scale is robust, with a fallback."""
    stats = {}
    for g, vals in goal_values.items():
        xs = list(vals.values())
        if not xs:
            continue
        med = _median(xs)
        scale = 1.4826 * _median([abs(x - med) for x in xs])
        if scale <= 0:
            scale = statistics.pstdev(xs)
        stats[g] = (med, scale)
    return stats


def _suspicion_from_z(z):
    return min(1.0, max(0.0, (abs(z) - _Z_LO) / _Z_SPAN))


def _goal_score(x, centre, scale):
    """Suspicion in [0, 1] for value x against a centre and scale."""
    if scale > 0:
        return _suspicion_from_z((x - centre) / scale)
    return 0.0 if x == centre else 1.0


def _ranked_neighbours(cols, train_rows, row):
    """(flag-set distance, training index) pairs, nearest first."""
    return sorted(
        (step2._distance(cols, tr, row), i) for i, tr in enumerate(train_rows)
    )


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included), in order.

    Higher means more likely a glitch, a typo, or a flag set unlike anything in
    train_csv. Scores lie in [0, 1.1]: the largest of the signals (each in [0, 1])
    plus a small share of their average.
    """
    cols, train_rows, goal_values, goal_names = step2._load(train_csv)
    if not rows:
        return []
    if not train_rows:
        return [1.0] * len(rows)  # nothing to compare against

    stats = _goal_stats(goal_values)
    scores = []
    for row in rows:
        ranked = _ranked_neighbours(cols, train_rows, row)

        nearest = ranked[0][0]
        parts = [min(1.0, nearest / _NOVEL_SCALE)]

        for g in goal_names:
            if g not in row or g not in stats:
                continue  # no evidence either way
            v = row[g]
            if not _finite(v):
                parts.append(_MISSING)
                continue

            med, scale = stats[g]
            parts.append(_goal_score(v, med, scale))

            twins = []
            for d, i in ranked:
                if d > _TWIN_DIST or len(twins) >= _TWINS:
                    break
                x = goal_values[g].get(i)
                if x is not None:
                    twins.append(x)
            if len(twins) >= 2:
                parts.append(_goal_score(v, _median(twins), scale))

        top = max(parts)
        mean = sum(parts) / len(parts)
        scores.append(float(top + _MIX * mean))
    return scores

