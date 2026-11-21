"""Step 1: flag-combination families and best rows for MOOT-format data."""

import csv
import random


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _load(path):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        rows = [{k: _num(v) for k, v in row.items()} for row in reader]
    return columns, rows


def _split_columns(columns):
    maximise, minimise, inputs = [], [], []
    for c in columns:
        if c.endswith("X"):
            continue
        if c.endswith("+"):
            maximise.append(c)
        elif c.endswith("-"):
            minimise.append(c)
        else:
            inputs.append(c)
    return maximise, minimise, inputs


def _encode_inputs(rows, inputs):
    vecs = [[0.0] * len(inputs) for _ in rows]
    for j, c in enumerate(inputs):
        numeric = all(isinstance(r.get(c), float) for r in rows)
        codes = {}
        for i, r in enumerate(rows):
            v = r.get(c)
            if numeric:
                vecs[i][j] = v
            else:
                key = str(v)
                if key not in codes:
                    codes[key] = len(codes)
                vecs[i][j] = float(codes[key])
    return vecs


def _sqdist(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _kmeans_once(X, k, rng, iters=100):
    n = len(X)
    dim = len(X[0])
    centers = [list(X[rng.randrange(n)])]
    d2 = [_sqdist(x, centers[0]) for x in X]
    while len(centers) < k:
        total = sum(d2)
        if total <= 0:
            idx = rng.randrange(n)
        else:
            target = rng.uniform(0, total)
            acc, idx = 0.0, n - 1
            for i, w in enumerate(d2):
                acc += w
                if acc >= target:
                    idx = i
                    break
        centers.append(list(X[idx]))
        d2 = [min(d2[i], _sqdist(X[i], centers[-1])) for i in range(n)]

    labels = None
    for _ in range(iters):
        new = []
        for x in X:
            dists = [_sqdist(x, c) for c in centers]
            new.append(dists.index(min(dists)))
        if new == labels:
            break
        labels = new
        for c in range(k):
            members = [X[i] for i in range(n) if labels[i] == c]
            if members:
                centers[c] = [
                    sum(m[j] for m in members) / len(members) for j in range(dim)
                ]

    sse = sum(_sqdist(X[i], centers[labels[i]]) for i in range(n))
    return labels, sse


def families(train_csv):
    """Return one family (cluster) label per data row of train_csv, in file order.

    Families are k-means clusters over the input (flag) columns only.
    """
    columns, rows = _load(train_csv)
    _, _, inputs = _split_columns(columns)
    n = len(rows)
    if n == 0:
        return []

    X = _encode_inputs(rows, inputs)
    k = min(5, n)
    rng = random.Random(0)

    best_labels, best_sse = None, None
    for _ in range(5):
        labels, sse = _kmeans_once(X, k, rng)
        if best_sse is None or sse < best_sse:
            best_labels, best_sse = labels, sse

    remap = {}
    out = []
    for lab in best_labels:
        if lab not in remap:
            remap[lab] = len(remap)
        out.append(remap[lab])
    return out


def _dominates(a, b):
    """True if a Pareto-dominates b (all dimensions minimised)."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def best(train_csv):
    """Return sorted row indices of the unusually good rows.

    Good means non-dominated on all goals together ('+' goals maximised,
    '-' goals minimised). At most 5% of rows are returned; if the Pareto
    front is larger, the rows with the best normalised goal sum are kept.
    """
    columns, rows = _load(train_csv)
    maximise, minimise, _ = _split_columns(columns)
    n = len(rows)
    cap = (n * 5) // 100
    if n == 0 or cap == 0 or not (maximise or minimise):
        return []

    # Everything becomes a minimisation problem.
    G = []
    for r in rows:
        G.append([-float(r[c]) for c in maximise] + [float(r[c]) for c in minimise])
    m = len(G[0])

    lo = [min(g[j] for g in G) for j in range(m)]
    hi = [max(g[j] for g in G) for j in range(m)]
    N = []
    for g in G:
        N.append([
            (g[j] - lo[j]) / (hi[j] - lo[j]) if hi[j] > lo[j] else 0.0
            for j in range(m)
        ])

    front = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if j != i and _dominates(N[j], N[i]):
                dominated = True
                break
        if not dominated:
            front.append(i)

    if len(front) > cap:
        front.sort(key=lambda i: (sum(N[i]), i))
        front = front[:cap]

    return sorted(front)

