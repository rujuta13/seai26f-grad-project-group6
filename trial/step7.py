"""step7.py: choose a good flag set while benchmarking only a few builds.

Standard library only (Python 3.13). Reuses step1 (csv and MOOT helpers,
input clustering with families(), input scaling, and goal normalisation).

The blind csv has no goal values, so nothing can be learned from it
directly. choose() spends its budget in two phases:
  1. Explore: benchmark builds spread across the input clusters from
     families(), farthest-first within each cluster.
  2. Exploit: repeatedly benchmark the unbenchmarked build with the best
     nearest-neighbour estimate, plus a small bonus for being far from
     everything benchmarked so far.
The answer is the benchmarked build with the best mean normalised score
over the goals in use (step1 scaling, 1 is best). If `goal` names one goal,
only that goal is optimised, so the speed-only and CPU-only questions are
answered by calling choose() with goal set to that goal.
"""
from __future__ import annotations

import math
import os

from step1 import (
    _classify,
    _encode_inputs,
    _goal_scores,
    _minmax,
    _read,
    _to_float,
    families,
)

_EXPLORE_SHARE = 0.5    # share of the budget spent on the spread-out phase
_MIN_EXPLORE = 3        # but never fewer than this many spread-out builds
_K = 3                  # neighbours used to estimate an unbenchmarked build
_EXPLORE_WEIGHT = 0.2   # bonus for builds far from everything benchmarked


def _select_goals(header: list[str], goals: list[tuple[int, int]],
                  goal: str | None) -> list[tuple[int, int]]:
    """All goals, or only the named one (with or without its +/- suffix)."""
    if goal is None:
        return goals
    for j, sign in goals:
        name = header[j]
        if name == goal or name[:-1] == goal:
            return [(j, sign)]
    names = [header[j] for j, _ in goals]
    raise ValueError(f"goal {goal!r} not found among {names}")


def _lookup(values: dict, name: str):
    """Value for a goal name, accepting the name with or without its suffix."""
    if name in values:
        return values[name]
    return values.get(name[:-1])


def _farthest(pool: list[int], X: list[list[float]], picked: list[int]) -> int:
    """Row in pool farthest from every picked row (first row if none picked)."""
    if not picked:
        return pool[0]
    return max(
        pool,
        key=lambda i: (min(math.dist(X[i], X[k]) for k in picked), -i),
    )


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best build found within `budget` calls to label(i).

    blind_csv: MOOT csv whose goal cells are all '?'.
    label(i): returns the goals of row i as a dict {goal column name: value}.
    budget: at most this many label calls (more raises RuntimeError).
    goal: optimise only this goal (name with or without +/- suffix).
    """
    path = os.fspath(blind_csv)
    header, rows = _read(path)
    inputs, goals = _classify(header)
    goals = _select_goals(header, goals, goal)
    n = len(rows)
    if n == 0:
        raise ValueError("blind_csv has no data rows")
    if not goals:
        raise ValueError("blind_csv has no goal columns")
    if budget < 1:
        raise ValueError("budget must be at least 1")
    budget = min(int(budget), n)

    X = _minmax(_encode_inputs(rows, inputs))

    observed: dict[int, dict[str, float]] = {}

    def probe(i: int) -> None:
        if i in observed:
            return
        if len(observed) >= budget:
            raise RuntimeError(f"label budget of {budget} calls exhausted")
        raw = label(i)
        vals: dict[str, float] = {}
        for j, _ in goals:
            name = header[j]
            v = _lookup(raw, name)
            if v is None:
                vals[name] = math.nan
            else:
                f = _to_float(str(v))
                vals[name] = math.nan if f is None else f
        observed[i] = vals

    def scores() -> dict[int, float]:
        """Mean normalised goal score of each benchmarked row (1 is best)."""
        ids = sorted(observed)
        table: list[list[str]] = []
        for i in ids:
            row = ["?"] * len(header)
            for j, _ in goals:
                row[j] = str(observed[i][header[j]])
            table.append(row)
        S = _goal_scores(table, goals)
        return {i: sum(s) / len(s) for i, s in zip(ids, S)}

    def estimate(i: int, known: list[int],
                 sc: dict[int, float]) -> tuple[float, float]:
        """Distance-weighted score of the nearest benchmarked rows, and the
        distance to the nearest one."""
        dists = sorted((math.dist(X[i], X[k]), k) for k in known)
        acc = 0.0
        wsum = 0.0
        for d, k in dists[:_K]:
            w = 1.0 / (d + 1e-9)
            wsum += w
            acc += w * sc[k]
        return acc / wsum, dists[0][0]

    # Phase 1: spread the first builds across the input clusters.
    clusters = families(path)
    groups: list[list[int]] = [[] for _ in range(max(clusters) + 1)]
    for i, c in enumerate(clusters):
        groups[c].append(i)
    pools = [g for g in groups if g]

    n_explore = min(budget, max(_MIN_EXPLORE, int(budget * _EXPLORE_SHARE)))
    picked: list[int] = []
    while len(picked) < n_explore and any(pools):
        for pool in pools:
            if not pool or len(picked) >= n_explore:
                continue
            pick = _farthest(pool, X, picked)
            pool.remove(pick)
            picked.append(pick)
    for i in picked:
        probe(i)

    # Phase 2: benchmark the most promising remaining builds.
    while len(observed) < budget:
        unknown = [i for i in range(n) if i not in observed]
        if not unknown:
            break
        known = sorted(observed)
        sc = scores()
        est: dict[int, float] = {}
        dmin: dict[int, float] = {}
        for i in unknown:
            est[i], dmin[i] = estimate(i, known, sc)
        top_d = max(dmin.values()) or 1.0
        nxt = max(
            unknown,
            key=lambda i: (est[i] + _EXPLORE_WEIGHT * dmin[i] / top_d, -i),
        )
        probe(nxt)

    sc = scores()
    return max(sc, key=lambda i: (sc[i], -i))

