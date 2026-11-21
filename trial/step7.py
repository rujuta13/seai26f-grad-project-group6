import numpy as np
from sklearn.neighbors import NearestNeighbors

from step1 import _read, _columns
from step2 import _encode_inputs

_K = 5
_EXPLORE = 0.5


def _float_or_nan(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def choose(blind_csv, label, budget=30, goal=None) -> int:
    """Pick the best row by labelling at most `budget` rows.

    Rows are described only by their inputs in `blind_csv`. Goals are obtained
    by calling label(i). A small diverse seed set is labelled first
    (farthest-point sampling over standardised inputs), then the rest of the
    budget goes to rows whose predicted quality (k-nearest-neighbour regression
    on labelled rows) plus a distance-based exploration bonus is highest.
    Returns the index of the best labelled row. If `goal` is given, only that
    goal is optimised; otherwise the min-max normalised goal sum is used.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")

    fieldnames, rows = _read(blind_csv)
    inputs, goals = _columns(fieldnames)
    n = len(rows)
    if n == 0:
        raise ValueError("blind_csv has no rows")
    if not goals:
        raise ValueError("blind_csv has no goal columns")

    names = [g for g, _ in goals]
    signs = np.array([s for _, s in goals], dtype=float)
    G = len(names)

    sel = None
    if goal is not None:
        if goal not in names:
            raise ValueError(f"unknown goal: {goal!r}")
        sel = names.index(goal)

    B = min(budget, n)
    if B == 0:
        return 0

    # Standardised input features for every row (goal cells are not used).
    if inputs:
        X, _ = _encode_inputs(rows, inputs)
    else:
        X = np.zeros((n, 1))
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd

    labeled: dict[int, np.ndarray] = {}
    calls = 0

    def probe(i):
        nonlocal calls
        if calls >= budget:
            raise RuntimeError("labelling budget exceeded")
        calls += 1
        d = label(i) or {}
        labeled[i] = np.array(
            [signs[j] * _float_or_nan(d.get(names[j])) for j in range(G)],
            dtype=float,
        )

    # Phase 1: diverse seed set by farthest-point sampling.
    n0 = min(B, max(2, B // 3))
    first = 0
    probe(first)
    dmin = np.linalg.norm(Z - Z[first], axis=1)
    while len(labeled) < n0:
        cand = dmin.copy()
        for i in labeled:
            cand[i] = -np.inf
        nxt = int(np.argmax(cand))
        probe(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(Z - Z[nxt], axis=1))

    # Phase 2: exploit predictions with an exploration bonus.
    while len(labeled) < B:
        Lidx = np.array(sorted(labeled), dtype=int)
        unl = np.array([i for i in range(n) if i not in labeled], dtype=int)
        if unl.size == 0:
            break

        vals = np.array([labeled[i] for i in Lidx], dtype=float)  # (L, G)
        P = np.zeros((len(unl), G))
        lo = np.zeros(G)
        hi = np.zeros(G)
        for g in range(G):
            mask = np.isfinite(vals[:, g])
            if not np.any(mask):
                continue
            li = Lidx[mask]
            yg = vals[mask, g]
            lo[g] = float(yg.min())
            hi[g] = float(yg.max())
            k = min(_K, len(li))
            nn = NearestNeighbors(n_neighbors=k).fit(Z[li])
            _, nb = nn.kneighbors(Z[unl])
            P[:, g] = yg[nb].mean(axis=1)

        span = np.where(hi - lo > 0, hi - lo, 1.0)
        if sel is not None:
            S = P[:, sel]
        else:
            S = ((P - lo) / span).sum(axis=1)
        s_span = S.max() - S.min()
        s = (S - S.min()) / (s_span if s_span > 0 else 1.0)

        dist, _ = NearestNeighbors(n_neighbors=1).fit(Z[Lidx]).kneighbors(Z[unl])
        D = dist[:, 0]
        Dmax = D.max()
        d_norm = D / Dmax if Dmax > 0 else np.zeros_like(D)

        acq = s + _EXPLORE * d_norm
        probe(int(unl[int(np.argmax(acq))]))

    # Final choice among labelled rows.
    idxs = np.array(sorted(labeled), dtype=int)
    M = np.array([labeled[i] for i in idxs], dtype=float)

    if sel is not None:
        ok = np.isfinite(M[:, sel])
        if not np.any(ok):
            return int(idxs[0])
        vals_sel = np.where(ok, M[:, sel], -np.inf)
        return int(idxs[int(np.argmax(vals_sel))])

    ok_rows = np.all(np.isfinite(M), axis=1)
    if not np.any(ok_rows):
        return int(idxs[0])
    M = M[ok_rows]
    idxs = idxs[ok_rows]
    lo = M.min(axis=0)
    hi = M.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    score = ((M - lo) / span).sum(axis=1)
    return int(idxs[int(np.argmax(score))])

