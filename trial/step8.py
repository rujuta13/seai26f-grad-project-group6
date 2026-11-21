"""step8.py: fewest input flips that make a bad build good.

fix() first looks for an observed build in the training data. It must be on the
training Pareto front (step1 dominance, the same test step4.stream and step6.why
use), agree with the query on every frozen column, and differ from it in the
fewest mutable input columns. Such a build has measured goals, so it is preferred.

If no observed build qualifies, or a predicted build needs fewer changes, fix()
runs a bounded breadth-first search over single-column changes. Each candidate's
goals are estimated with step2.guess, and a candidate counts as good when that
estimate is not dominated by any training build. Predicted results are used only
when they need strictly fewer changes than the observed build, so ties go to the
measured build.

Loading, classification, goal matrices, value coercion and the mean-rank helper
are reused from step1 and step6. guess() comes from step2.
"""
from __future__ import annotations

import numpy as np

from step1 import _classify, _goal_matrix, _load, _to_value
from step2 import guess
from step6 import _mean_rank

_BUDGET = 400        # maximum number of predicted (guessed) candidates evaluated
_BEAM = 20           # predicted candidates carried into the next search level
_MAX_VALUES = 12     # values tried per mutable column (evenly spaced if more)


def _norm(v):
    """Comparable form of a cell: float if numeric, otherwise its string."""
    if isinstance(v, bool):
        return str(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def _is_dominated(G, g):
    """True if some row of G is at least as good on every goal and strictly better on one."""
    return bool(((G <= g).all(axis=1) & (G < g).any(axis=1)).any())


def _domain(rows, col):
    """Distinct raw cell values seen in the training data for one input column."""
    first = {}
    for r in rows:
        first.setdefault(_norm(r[col]), r[col])
    vals = sorted(first.values(), key=lambda v: (isinstance(v, str), _norm(v)))
    if len(vals) > _MAX_VALUES:
        idx = np.unique(np.linspace(0, len(vals) - 1, _MAX_VALUES).round().astype(int))
        vals = [vals[i] for i in idx]
    return vals


def _render(cfg, base, names):
    """Output dict: unchanged columns keep the caller's value, changed ones are coerced."""
    out = {}
    for n in names:
        if _norm(cfg[n]) == _norm(base[n]):
            out[n] = base[n]
        else:
            v = cfg[n]
            out[n] = _to_value(v) if isinstance(v, str) else v
    return out


def fix(train_csv, row, frozen=()) -> dict:
    """Inputs of `row` with as few values changed as possible so the build becomes good.

    row: dict {input column name: value} for a bad build (goals removed).
    frozen: input column names that must not change.
    Returns {input column name: value} for every input column of the training file.
    If no candidate is found within the search budget, the lowest-scoring predicted
    configuration is returned, and it is not guaranteed to be good.
    """
    header, rows = _load(train_csv)
    if not rows:
        raise ValueError("train_csv contains no data rows")
    maximise, minimise, inputs = _classify(header)
    names = [header[c] for c in inputs]

    frozen = set(frozen)
    unknown = frozen - set(names)
    if unknown:
        raise ValueError(f"frozen columns are not input columns: {sorted(unknown)}")
    missing = [n for n in names if n not in row]
    if missing:
        raise KeyError(f"row is missing input columns: {missing}")

    base = {n: row[n] for n in names}
    mutable = [n for n in names if n not in frozen]
    G = _goal_matrix(rows, maximise, minimise)
    if G.shape[1] == 0:
        raise ValueError("train_csv has no goal columns")

    def changes(cfg):
        return sum(_norm(cfg[n]) != _norm(base[n]) for n in names)

    # Observed candidates: training builds on the Pareto front that agree on frozen columns.
    rank = _mean_rank(G)
    best_obs = None
    for i in range(len(rows)):
        if _is_dominated(G, G[i]):
            continue
        cand = {n: rows[i][header[inputs[names.index(n)]]] for n in names}
        if any(_norm(cand[n]) != _norm(base[n]) for n in frozen):
            continue
        key = (changes(cand), float(rank[i]))
        if best_obs is None or key < best_obs[0]:
            best_obs = (key, cand)

    if best_obs is not None and best_obs[0][0] == 0:
        return _render(base, base, names)

    # Predicted search, limited to fewer changes than the observed build if one exists.
    max_level = len(mutable) if best_obs is None else best_obs[0][0] - 1
    search_cfg = _search(train_csv, header, rows, G, maximise, minimise,
                         names, mutable, base, max_level, changes)

    if search_cfg is not None:
        return _render(search_cfg, base, names)
    if best_obs is not None:
        return _render(best_obs[1], base, names)
    return _render(_best_effort_last, base, names) if _best_effort_last is not None else _render(base, base, names)


_best_effort_last = None  # set by _search when nothing good is found


def _search(train_csv, header, rows, G, maximise, minimise, names, mutable, base,
            max_level, changes):
    """Breadth-first search over single-column changes using step2.guess predictions.

    Returns the good configuration with the fewest changes (ties broken by predicted
    score), or None. Sets _best_effort_last to the best-scoring configuration seen.
    """
    global _best_effort_last
    _best_effort_last = None
    if max_level < 1 or not mutable:
        return None

    mu = G.mean(axis=0)
    sd = G.std(axis=0) + 1e-12
    domains = {n: _domain(rows, inputs_col) for n, inputs_col in
               ((n, header.index(n)) for n in mutable)}

    def predict(cfg):
        est = guess(train_csv, cfg)
        gv = np.array([est[header[c]][0] for c in minimise]
                      + [-est[header[c]][0] for c in maximise], dtype=float)
        good = not _is_dominated(G, gv)
        score = float((((gv - mu) / sd)).sum())
        return good, score

    seen = {tuple(_norm(base[n]) for n in names)}
    frontier = [(base, 0.0)]
    best_effort = None  # (score, changes, cfg)
    evals = 0

    for _level in range(1, max_level + 1):
        found = []
        scored = []
        for cfg, _ in frontier:
            for n in mutable:
                for v in domains[n]:
                    if _norm(v) == _norm(cfg[n]):
                        continue
                    new = dict(cfg)
                    new[n] = v
                    key = tuple(_norm(new[m]) for m in names)
                    if key in seen:
                        continue
                    seen.add(key)
                    if evals >= _BUDGET:
                        _best_effort_last = best_effort[2] if best_effort else None
                        return _pick(found)
                    evals += 1
                    good, score = predict(new)
                    k = changes(new)
                    if best_effort is None or (score, k) < (best_effort[0], best_effort[1]):
                        best_effort = (score, k, new)
                    if good:
                        found.append((k, score, new))
                    scored.append((score, new))
        if found:
            _best_effort_last = best_effort[2] if best_effort else None
            return _pick(found)
        scored.sort(key=lambda t: t[0])
        frontier = [(cfg, s) for s, cfg in scored[:_BEAM]]
        if not frontier:
            break

    _best_effort_last = best_effort[2] if best_effort else None
    return None


def _pick(found):
    """Good configuration with the fewest changes, then the best predicted score."""
    if not found:
        return None
    found.sort(key=lambda t: (t[0], t[1]))
    return found[0][2]

