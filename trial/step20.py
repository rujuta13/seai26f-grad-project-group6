# step20.py
"""Step 20: race.  Does our shortlist beat a simple "top overall rating"
list, and does our penalty guesser beat an off-the-shelf model?

Every score is lower-is-better.  Shortlist scores are the mean ydist
(distance from the ideal over all goals, 0 is best) of the builds each
method puts on its shortlist from TEST_CSV.  Guesser scores are the mean
normalised error of the Penalties+ guess over test rows (0 is best).
The keys are:
  "ours"          our shortlist: the test rows whose flags look nearest
                  the best quarter of training rows.
  "top_overall"   the simple list: the highest OVR rows.
  "random_search" a random shortlist, averaged over many draws.
  "guess_ours"    our penalty guesser: median Penalties+ of the nearest
                  training rows.
  "linear_ridge"  an off-the-shelf linear regression for Penalties+.

Uses only the standard library plus ezr.
"""
import math
import random
import ezr

_SHORT    = 0.10     # shortlist size, as a share of the test rows
_DRAWS    = 20       # random shortlists averaged for random_search
_RIDGE    = 1e-3     # ridge penalty for the linear rival
_MAX_LVLS = 25       # linear rival: one-hot only symbols with at most this many levels
_DROP     = {"num", "Rank", "Name", "Nation", "League", "Team", "url"}
_KEYS     = ("ours", "top_overall", "random_search", "guess_ours", "linear_ridge")


def _num(v) -> bool:
  "True if V is a usable number (not text, not missing, not a bool)."
  return isinstance(v, (int, float)) and not isinstance(v, bool)


def _stem(name: str) -> str:
  "A column name without its goal sign, lower-cased: 'Penalties+' -> 'penalties'."
  return name.rstrip("+-!").strip().lower()


def _clean(v):
  "A cell as ezr reads it: strings are cast, blanks and None become '?'."
  if v is None: return "?"
  if isinstance(v, str):
    s = v.strip()
    return ezr.atom(s) if s else "?"
  return v


def _median(xs: list[float]) -> float:
  "Median of XS, interpolated between the middle two when even; XS must be non-empty."
  xs = sorted(xs)
  m  = len(xs)
  return xs[m // 2] if m % 2 else (xs[m // 2 - 1] + xs[m // 2]) / 2


def _mean(xs: list[float]) -> float:
  "Mean of XS, or infinity if there is nothing to score."
  return sum(xs) / len(xs) if xs else math.inf


def _load(path: str):
  "Read PATH: (header names, rows padded to full width, cells cleaned)."
  raw = list(ezr.csv(path))
  if not raw: return [], []
  names = [str(h).strip() for h in raw[0]]
  rows  = []
  for r in raw[1:]:
    r = [_clean(v) for v in r][:len(names)]
    rows.append(r + ["?"] * (len(names) - len(r)))
  return names, rows


def _align(names: list, src_names: list, src_rows: list) -> list:
  "Rows of SRC laid out in the column order of NAMES; a missing column is '?'."
  pos = {}
  for at, nm in enumerate(src_names): pos.setdefault(nm, at)
  idx = [pos.get(nm) for nm in names]
  return [[r[i] if i is not None and i < len(r) else "?" for i in idx]
          for r in src_rows]


def _is_num(rows: list, at: int) -> bool:
  "True if most known cells of column AT are numbers."
  known = [r[at] for r in rows if r[at] != "?"]
  nums  = sum(1 for v in known if _num(v))
  return nums > 0 and 2 * nums >= len(known)


def _model(names: list, rows: list, at: int) -> ezr.Num:
  "An ezr.Num summarising the numeric cells of column AT; its goal sign comes from its name."
  return ezr.adds([r[at] for r in rows if _num(r[at])],
                  ezr.Num(names[at], at))


def _layout(names: list, rows: list):
  "(goals, inputs) of the kept columns: goals are (at, Num); inputs are (at, Num or None if symbolic)."
  goals, inputs = [], []
  for at, nm in enumerate(names):
    if not nm or nm in _DROP or nm.endswith("X"): continue
    numeric = _is_num(rows, at)
    if nm[-1] in "+-":
      if numeric: goals.append((at, _model(names, rows, at)))
    else:
      inputs.append((at, _model(names, rows, at) if numeric else None))
  return goals, inputs


def _ydist(goals: list, row: list) -> float:
  "How far ROW's goals are from the ideal, RMS over all goals; a missing goal is worst."
  gaps = [abs(ezr.norm(m, row[at]) - m.goal) if _num(row[at]) else 1.0
          for at, m in goals]
  return ezr.dist(gaps, len(gaps)) if gaps else 0.0


def _enc(m, v):
  "An input cell for distance: None if unknown, symbols as-is, numbers scaled to 0..1."
  if m is None: return None if v == "?" else v
  return ezr.norm(m, v) if _num(v) else None


def _dist(sym: list[bool], a: list, b: list) -> float:
  "Root mean square gap between two encoded input rows, as ezr.xdist."
  if not sym: return 0.0
  s = 0.0
  for f, x, y in zip(sym, a, b):
    if x is None or y is None: g = 1.0
    elif f:                    g = 0.0 if x == y else 1.0
    else:                      g = abs(x - y)
    s += g * g
  return math.sqrt(s / len(sym))


def _rank_key(v, t: int):
  "Sort key: highest numbers first, unknown last, ties by row order."
  return (0, -v, t) if _num(v) else (1, 0, t)


def _design(inp: list, tr_rows: list):
  "Feature maker for the linear rival: bias, standardised numbers, one-hot symbols."
  spec = []
  for at, m in inp:
    if m is not None:
      spec.append(("num", at, m))
    else:
      lv = sorted({r[at] for r in tr_rows if r[at] != "?"}, key=str)
      if 0 < len(lv) <= _MAX_LVLS:
        spec.append(("sym", at, lv))

  def feats(row: list) -> list[float]:
    "One row as numbers: a bias, standardised numbers (0 if unknown), one-hot symbols."
    out = [1.0]
    for kind, at, d in spec:
      v = row[at]
      if kind == "num":
        out.append((v - d.mu) / (d.sd or 1.0) if _num(v) else 0.0)
      else:
        out += [1.0 if v == s else 0.0 for s in d]
    return out

  return feats


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
  "Solve A w = b by Gauss-Jordan elimination with partial pivoting."
  n = len(b)
  M = [A[i][:] + [b[i]] for i in range(n)]
  for col in range(n):
    piv = max(range(col, n), key=lambda r: abs(M[r][col]))
    M[col], M[piv] = M[piv], M[col]
    if abs(M[col][col]) < 1e-12: continue
    for r in range(n):
      if r != col and M[r][col] != 0.0:
        f = M[r][col] / M[col][col]
        M[r] = [a - f * c for a, c in zip(M[r], M[col])]
  return [M[i][n] / M[i][i] if abs(M[i][i]) >= 1e-12 else 0.0
          for i in range(n)]


def _fit(X: list[list[float]], y: list[float]) -> list[float]:
  "Ridge least squares: weights w minimising |Xw - y|^2 + RIDGE |w|^2."
  p = len(X[0])
  A = [[0.0] * p for _ in range(p)]
  b = [0.0] * p
  for x, t in zip(X, y):
    for i in range(p):
      xi = x[i]
      b[i] += xi * t
      Ai = A[i]
      for j in range(p):
        Ai[j] += xi * x[j]
  for i in range(p): A[i][i] += _RIDGE
  return _solve(A, b)


def race(train_csv: str, test_csv: str) -> dict[str, float]:
  """Score each method on TEST_CSV, lower is better.  The shortlists pick
  test builds using the flags and the training data, and are scored by the
  true distance from the ideal of the builds picked.  The penalty guessers
  are scored on how close their Penalties+ guess lands to the true value.
  Always includes the key "ours"."""
  out = {k: math.inf for k in _KEYS}
  names, tr_raw = _load(train_csv)
  tnames, te_raw = _load(test_csv)
  if not tr_raw or not te_raw: return out
  goals, inp = _layout(names, tr_raw)
  if not goals: return out

  te   = _align(names, tnames, te_raw)
  n, m = len(tr_raw), len(te)
  ty   = [_ydist(goals, r) for r in tr_raw]       # training goal distances
  ey   = [_ydist(goals, r) for r in te]           # truth on the test rows
  sym  = [mo is None for _, mo in inp]
  E    = [[_enc(mo, r[at]) for at, mo in inp] for r in tr_raw]
  T    = [[_enc(mo, r[at]) for at, mo in inp] for r in te]
  k    = min(n, max(5, int(math.sqrt(n))))

  near = []                                       # nearest training rows, per test row
  for t in T:
    d = [_dist(sym, t, e) for e in E]
    near.append(sorted(range(n), key=lambda i: (d[i], i))[:k])

  size = max(1, int(_SHORT * m))

  # Ours: shortlist the test builds whose nearest training builds are best.
  pred  = [_median([ty[i] for i in near[t]]) for t in range(m)]
  order = sorted(range(m), key=lambda t: (pred[t], t))
  out["ours"] = _mean([ey[t] for t in order[:size]])

  # The simple rival: the highest overall rating, OVR.
  ovr = next((at for at, nm in enumerate(names) if _stem(nm) == "ovr"), None)
  if ovr is not None:
    order = sorted(range(m), key=lambda t: _rank_key(te[t][ovr], t))
    out["top_overall"] = _mean([ey[t] for t in order[:size]])

  # Random search: the expected quality of a random shortlist.
  rng = random.Random(ezr.the.Seed)
  out["random_search"] = _mean([
      _mean([ey[t] for t in rng.sample(range(m), min(size, m))])
      for _ in range(_DRAWS)])

  # Penalty guesser versus an off-the-shelf linear model, on Penalties+.
  pen = next(((at, mo) for at, mo in goals
              if _stem(names[at]) == "penalties"), None)
  if pen is not None:
    pat, pm  = pen
    known_tr = [i for i in range(n) if _num(tr_raw[i][pat])]
    if known_tr:
      tv    = {i: tr_raw[i][pat] for i in known_tr}
      all_v = list(tv.values())
      feats = _design(inp, tr_raw)
      X     = [feats(tr_raw[i]) for i in known_tr]
      y     = [ezr.norm(pm, tv[i]) for i in known_tr]
      w     = _fit(X, y)
      errs, lin_errs = [], []
      for t in range(m):
        actual = te[t][pat]
        if not _num(actual): continue
        truth = ezr.norm(pm, actual)
        vs    = [tv[i] for i in near[t] if i in tv] or all_v
        errs.append(abs(ezr.norm(pm, _median(vs)) - truth))
        lin = sum(a * b for a, b in zip(w, feats(te[t])))
        lin_errs.append(abs(lin - truth))
      out["guess_ours"]   = _mean(errs)
      out["linear_ridge"] = _mean(lin_errs)

  return out

