"""Step 7: choose. Find a good flag set while benchmarking only a limited number of builds.

Method: an adaptive search that labels rows one at a time through `label`.
  1. Explore: spend about 40% of the budget on rows spread across the input
     space (farthest-point sampling over step1's flag features), so the search
     sees the whole landscape before committing.
  2. Exploit: for each unlabelled candidate, predict its goal total from the
     nearest measured rows (distance-weighted k-nearest-neighbour), add a bonus
     for being far from anything measured, and label the candidate with the
     highest combined score. Each round labels exactly one new row.

Goals are direction-aware and min-max normalised over the measured rows; a
missing value counts as worst. With goal=None all goals are summed (as in
step1.best). With one goal named, only that goal is optimised. The returned
index is the best measured row, ties going to the lower index.
"""

import heapq
import math
import random

import step1

_SEED = 12345            # fixed seed so results are reproducible
_EXPLORE_FRACTION = 0.4  # share of the budget spent on spread-out exploration
_NEIGHBOURS = 5          # measured neighbours used for each prediction
_KAPPA = 0.5             # weight of the distance (uncertainty) bonus
_EPS = 1e-9


def _resolve_goals(goal, goals_max, goals_min):
    """Return [(goal name, sign)] to optimise. A bare name such as 'Cpu' matches 'Cpu-'."""
    every = [(g, 1.0) for g in goals_max] + [(g, -1.0) for g in goals_min]
    if goal is None:
        return every
    for g, sign in every:
        if g == goal:
            return [(g, sign)]
    matches = [(g, sign) for g, sign in every if g[:-1] == goal]
    if len(matches) == 1:
        return matches
    raise ValueError(f"goal {goal!r} is not a goal column; goals are "
                     f"{[g for g, _ in every]}")


def _num(v):
    """Finite float for v, or None if v is missing or not a number."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _totals(measured, goals):
    """Normalised goal total per measured row (higher is better).

    measured: {row index: {goal: raw value or None}}. Each goal is signed so
    that higher is better, scaled to [0, 1] over the measured rows, and summed.
    Missing values count as 0 (worst).
    """
    totals = {i: 0.0 for i in measured}
    for g, sign in goals:
        present = [sign * d[g] for d in measured.values() if d[g] is not None]
        if not present:
            continue
        lo, hi = min(present), max(present)
        span = hi - lo
        for i, d in measured.items():
            if d[g] is None:
                continue
            x = sign * d[g]
            totals[i] += (x - lo) / span if span > 0 else 1.0
    return totals


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best row found, labelling at most `budget` rows.

    blind_csv: MOOT csv whose goal cells are '?'; only its input columns are used.
    label(i): returns the goals of row i as a dict {goal: value}. It is called at
        most `budget` times (a row is never labelled twice).
    goal: optional goal name (e.g. 'Performance-' or 'Performance'); when given,
        only that goal is optimised.
    """
    header, rows = step1._read(blind_csv)
    n = len(rows)
    if n == 0:
        raise ValueError("blind_csv has no data rows")
    if budget < 1:
        raise ValueError("budget must be at least 1")
    inputs, goals_max, goals_min = step1._classify(header)
    goals = _resolve_goals(goal, goals_max, goals_min)
    if not goals:
        raise ValueError("blind_csv has no goal columns (names ending in + or -)")

    feats = step1._features(rows, inputs)
    budget = min(budget, n)

    measured = {}                      # row index -> {goal: value or None}
    unlabelled = set(range(n))
    dmin = [math.inf] * n              # squared distance to nearest measured row

    def measure(i):
        if len(measured) >= budget:
            raise RuntimeError("label budget exhausted")
        got = label(i) or {}
        measured[i] = {g: _num(got.get(g)) for g, _ in goals}
        unlabelled.discard(i)
        fi = feats[i]
        for j in range(n):
            d = step1._dist2(feats[j], fi)
            if d < dmin[j]:
                dmin[j] = d

    # Phase 1: spread-out exploration.
    rng = random.Random(_SEED)
    n_explore = min(budget, max(1, round(budget * _EXPLORE_FRACTION)))
    measure(rng.randrange(n))
    while len(measured) < n_explore and unlabelled:
        j = max(sorted(unlabelled), key=lambda k: dmin[k])
        measure(j)

    # Phase 2: predict-and-probe, one row per round.
    while len(measured) < budget and unlabelled:
        totals = _totals(measured, goals)
        vals = list(totals.values())
        spread = (max(vals) - min(vals)) if max(vals) > min(vals) else 1.0

        cand = []  # (predicted total, distance to measured set, row index)
        for j in sorted(unlabelled):
            near = heapq.nsmallest(
                _NEIGHBOURS,
                ((step1._dist2(feats[j], feats[i]), i) for i in measured),
            )
            wsum = sum(1.0 / (d + _EPS) for d, _ in near)
            pred = sum(totals[i] / (d + _EPS) for d, i in near) / wsum
            cand.append((pred, math.sqrt(dmin[j]), j))

        dmax = max(u for _, u, _ in cand) or 1.0
        best_j, best_score = None, -math.inf
        for pred, unc, j in cand:
            score = pred + _KAPPA * spread * unc / dmax
            if score > best_score:
                best_j, best_score = j, score
        measure(best_j)

    totals = _totals(measured, goals)
    return min(measured, key=lambda i: (-totals[i], i))

