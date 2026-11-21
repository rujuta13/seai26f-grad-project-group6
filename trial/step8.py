import itertools

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from step1 import _column_kind, _goal_matrix, _load
from step2 import _build_encoders, _encode


_N_TREES = 200
_MAX_CANDIDATES = 20000   # cap on total candidate rows scored per call
_CHUNK = 256              # rows per dominance-check block


def _frozen_set(frozen):
    if frozen is None:
        return set()
    if isinstance(frozen, str):
        return {frozen}
    return set(frozen)


def fix(train_csv, row, frozen=()) -> dict:
    """Return a copy of `row` (inputs only changed) that is predicted to be good,
    changing as few input values as possible. Columns named in `frozen` are
    never changed.

    Goals are predicted for candidate inputs with random forests fitted on the
    training builds. A candidate is good when no training build is at least as
    good on every goal and strictly better on one (the same test as step4.stream).
    Among good candidates with the fewest changes, the one with the best
    predicted goals is returned. If no candidate is predicted good within the
    search budget, the candidate with the fewest dominating training builds is
    returned instead.
    """
    header, rows = _load(train_csv)
    frozen = _frozen_set(frozen)
    out = dict(row)

    goals = [c for c in header if _column_kind(c) in ("max", "min")]
    inputs = [c for c in header if _column_kind(c) == "input"]
    n = len(rows)
    if n < 4 or not goals or not inputs:
        return out

    G = _goal_matrix(header, rows)  # (n, g), smaller is always better
    values_by_col = {c: [r.get(c) for r in rows] for c in inputs}
    encoders = _build_encoders(values_by_col)

    X = np.zeros((n, len(inputs)), dtype=float)
    for j, c in enumerate(inputs):
        X[:, j] = [_encode(v, encoders[c]) for v in values_by_col[c]]
    base = np.array(
        [_encode(row.get(c), encoders[c]) for c in inputs], dtype=float
    )

    models = [
        RandomForestRegressor(
            n_estimators=_N_TREES,
            min_samples_leaf=2,
            random_state=0,
            n_jobs=1,
        ).fit(X, G[:, g])
        for g in range(G.shape[1])
    ]

    mu = G.mean(axis=0)
    sd = G.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)

    def evaluate(Xc):
        P = np.column_stack([m.predict(Xc) for m in models])
        counts = np.zeros(len(Xc), dtype=int)
        for s in range(0, len(Xc), _CHUNK):
            Pc = P[s:s + _CHUNK]
            le = (G[None, :, :] <= Pc[:, None, :]).all(axis=2)
            lt = (G[None, :, :] < Pc[:, None, :]).any(axis=2)
            counts[s:s + _CHUNK] = (le & lt).sum(axis=1)
        score = ((P - mu) / sd).sum(axis=1)
        return counts, score

    # Candidate values per input column: other values seen in training,
    # keyed by their encoded value so each alternative appears once.
    domains = {}
    for c in inputs:
        seen = {}
        for v in values_by_col[c]:
            seen.setdefault(str(v), v)
        domains[c] = list(seen.values())

    changeable = []  # (column index, [(encoded, raw), ...])
    for j, c in enumerate(inputs):
        if c in frozen:
            continue
        alts = {}
        for v in domains[c]:
            if str(v) == str(row.get(c)):
                continue
            enc = float(_encode(v, encoders[c]))
            if enc == float(base[j]):
                continue
            alts.setdefault(enc, (enc, v))
        if alts:
            changeable.append((j, list(alts.values())))

    good_rec = None   # (score, cols, raws)
    any_rec = None    # ((count, k, score), cols, raws)
    evaluated = 0
    stop = False

    for k in range(len(changeable) + 1):
        level_good = None
        for combo in itertools.combinations(changeable, k):
            cols = [j for j, _ in combo]
            combos = list(itertools.product(*[alts for _, alts in combo])) if k else [()]
            m = len(combos)
            if evaluated + m > _MAX_CANDIDATES:
                stop = True
                break
            evaluated += m

            Xc = np.tile(base, (m, 1))
            if k:
                Xc[:, cols] = np.array(
                    [[enc for enc, _ in cmb] for cmb in combos], dtype=float
                )
            counts, score = evaluate(Xc)

            for i in range(m):
                raws = tuple(raw for _, raw in combos[i]) if k else ()
                s = float(score[i])
                if counts[i] == 0:
                    cand = (s, cols, raws)
                    if level_good is None or cand[0] < level_good[0]:
                        level_good = cand
                key = (int(counts[i]), k, s)
                if any_rec is None or key < any_rec[0]:
                    any_rec = (key, cols, raws)

        if level_good is not None:
            good_rec = level_good
            break
        if stop:
            break

    if good_rec is not None:
        _, cols, raws = good_rec
    elif any_rec is not None:
        _, cols, raws = any_rec
    else:
        return out

    for j, raw in zip(cols, raws):
        out[inputs[j]] = raw
    return out

