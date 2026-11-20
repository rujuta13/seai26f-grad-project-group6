"""step7.py: choose. Find a good flag set while benchmarking at most `budget` builds.

blind_csv has the input cells of every candidate build, and every goal cell is '?'.
Goals are only learned by calling label(i), which may be called at most `budget`
times. The search spreads a first pass over the input space (farthest-point
sampling), then alternates between refining near the current leader and
exploring again. The leader is chosen with the same Pareto-then-mean-score rule
as step1.best, applied to the builds measured so far. With `goal` set, only that
goal is optimised.

Reading, column splitting, feature scaling, distances, the seed, number checks and
the dominance test come from steps 1 and 4.
"""
from __future__ import annotations

import math
import random

from step1 import _SEED, _columns, _dist2, _features, _num, _read
from step4 import _dominates

EXPLORE_SHARE = 0.5     # share of the budget spent on the first, spread-out pass
EXPLOIT_EVERY = 3       # every third follow-up benchmark explores instead of refining


def _value(v) -> float | None:
    """Return v as a finite float, or None if it is not a usable number."""
    if isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return _num(x)


def _resolve_goal(goal, goals: list[str]) -> str:
    """Accept a full goal column name, or the name without its trailing + or -."""
    if goal in goals:
        return goal
    hits = [g for g in goals if g[:-1] == goal and g[-1:] in ("+", "-")]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"unknown goal {goal!r}; goals are {goals}")


def _signed(got: dict, active: list[str], sign: list[float]) -> list[float]:
    """Signed goal vector for one measured build: larger is better on every axis.
    A missing or non-numeric goal counts as the worst possible value."""
    out = []
    for g, s in zip(active, sign):
        v = _value(got.get(g))
        out.append(s * v if v is not None else -math.inf)
    return out


def _scores(obs: dict[int, list[float]], m: int) -> dict[int, float]:
    """Mean min-max-normalised score per measured build, bounds taken from the
    measured builds only. Missing values score 0, as in step1.best."""
    norm = {i: [0.0] * m for i in obs}
    for j in range(m):
        col = [(i, v[j]) for i, v in obs.items() if math.isfinite(v[j])]
        if not col:
            continue
        xs = [x for _, x in col]
        lo, hi = min(xs), max(xs)
        span = hi - lo
        for i, x in col:
            norm[i][j] = (x - lo) / span if span > 0 else 1.0
    return {i: sum(v) / m for i, v in norm.items()}


def _leader(obs: dict[int, list[float]], m: int) -> int:
    """Best measured build: the top of the Pareto front over the measured builds,
    ordered by mean normalised score. With one goal this is the best value."""
    score = _scores(obs, m)
    front = [
        i for i in obs
        if not any(_dominates(obs[j], obs[i]) for j in obs if j != i)
    ]
    return min(front, key=lambda i: (-score[i], i))


def _nearest(target: int, candidates: list[int], feats: list[list[float]]) -> int:
    """Unmeasured candidate whose inputs are closest to the target's inputs."""
    return min(candidates, key=lambda i: (_dist2(feats[i], feats[target]), i))


def _farthest(candidates: list[int], measured: list[int],
              feats: list[list[float]]) -> int:
    """Unmeasured candidate whose inputs are farthest from every measured build."""
    return max(
        candidates,
        key=lambda i: (min(_dist2(feats[i], feats[j]) for j in measured), -i),
    )


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Index of the best build found while calling label at most `budget` times.

    blind_csv: a MOOT csv whose goal cells are all '?'.
    label(i): returns the goals of row i as a dict {goal column: value}.
    budget: the most label calls allowed. Going over raises RuntimeError.
    goal: optional goal column name (or its name without the trailing + or -).
          When given, only that goal is optimised. When None, all goals are
          optimised by Pareto fronts, as in step1.best.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a positive integer")

    header, rows = _read(str(blind_csv))
    inputs, gmax, gmin = _columns(header)
    goals = gmax + gmin
    n = len(rows)
    if n == 0:
        raise ValueError("blind_csv has no rows")
    if not goals:
        raise ValueError("blind_csv has no goal columns")

    if goal is None:
        active = list(goals)
    else:
        active = [_resolve_goal(goal, goals)]
    m = len(active)
    sign = [1.0 if g in gmax else -1.0 for g in active]

    limit = min(budget, n)
    feats = _features(rows, inputs)

    obs: dict[int, list[float]] = {}
    calls = 0

    def measure(i: int) -> None:
        nonlocal calls
        if calls >= budget:
            raise RuntimeError("label budget exhausted")
        calls += 1
        got = label(i) or {}
        obs[i] = _signed(got, active, sign)

    # First pass: spread the measurements over the input space.
    explore_n = min(limit, max(1, math.ceil(limit * EXPLORE_SHARE)))
    rng = random.Random(_SEED)
    measure(rng.randrange(n))
    while len(obs) < explore_n:
        rest = [i for i in range(n) if i not in obs]
        if not rest:
            break
        measure(_farthest(rest, list(obs), feats))

    # Remaining budget: mostly refine near the current leader, sometimes explore.
    step = 0
    while len(obs) < limit:
        rest = [i for i in range(n) if i not in obs]
        if not rest:
            break
        step += 1
        if step % EXPLOIT_EVERY == 0:
            nxt = _farthest(rest, list(obs), feats)
        else:
            nxt = _nearest(_leader(obs, m), rest, feats)
        measure(nxt)

    return _leader(obs, m)

