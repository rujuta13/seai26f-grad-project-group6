# step8.py
"Step 8: fix. The fewest input changes that make a bad build good, leaving frozen columns alone."
import ezr

QUANTILE = 0.25   # a build is "good" if its goals are in the best 25% of train
K        = 20     # nearest training rows consulted to predict a build's goals

_CACHE: dict = {}

def _coerce(v):
  "Strings become atoms; other values pass through."
  return ezr.atom(v) if isinstance(v, str) else v

def _full(tbl, row) -> list:
  """Lift ROW into a full-width row. Accepts a {column name: value} dict,
  a full-width list, or a list with goal columns removed (header order).
  Any cell not given is '?'."""
  names = tbl.names
  out   = ["?"] * len(names)
  if isinstance(row, dict):
    for at, n in enumerate(names):
      if n in row: out[at] = _coerce(row[n])
    return out
  row = list(row)
  if len(row) == len(names):
    return [_coerce(v) for v in row]
  keep = [i for i, n in enumerate(names) if n[-1] not in "+-!"]
  for i, v in zip(keep, row): out[i] = _coerce(v)
  return out

def _ydist(tbl, row) -> float | None:
  "Distance of ROW's numeric goals from their best; None if no goal is known."
  sq, n = 0.0, 0
  for c in tbl.y:
    if type(c) is not ezr.Num: continue
    v = row[c.at]
    if v == "?": continue
    sq += (ezr.norm(c, v) - c.goal) ** 2
    n  += 1
  return (sq / n) ** .5 if n else None

def _load(train_csv: str):
  "Train table and the distance cutoff for good builds; cached per file."
  if train_csv not in _CACHE:
    tbl = ezr.Tbl(ezr.csv(train_csv))
    ys  = sorted(d for d in (_ydist(tbl, r) for r in tbl.rows)
                 if d is not None)
    thr = ys[int(QUANTILE * (len(ys) - 1))] if ys else float("inf")
    _CACHE[train_csv] = (tbl, thr)
  return _CACHE[train_csv]

def _predict(tbl, q: list) -> float | None:
  """Predicted goal distance of input vector Q: the median goal value of
  the K training rows nearest to Q over the input columns."""
  near  = sorted(tbl.rows, key=lambda r: ezr.xdist(tbl, q, r))[:K]
  guess = ["?"] * len(q)
  for c in tbl.y:
    if type(c) is not ezr.Num: continue
    xs = sorted(r[c.at] for r in near if r[c.at] != "?")
    if xs: guess[c.at] = xs[len(xs) // 2]
  return _ydist(tbl, guess)

def _inputs(tbl, v: list) -> dict:
  "Input columns of full-width row V, by name."
  return {c.txt: v[c.at] for c in tbl.x}

def fix(train_csv, row, frozen=()) -> dict:
  """Inputs of bad ROW with as few values changed as possible so that it
  becomes good. Columns named in FROZEN never change. Returns
  {input column name: value}."""
  tbl, thr = _load(train_csv)
  if isinstance(frozen, str): frozen = (frozen,)
  frozen_set = set(frozen)
  free = [c for c in tbl.x if c.txt not in frozen_set]
  q    = _full(tbl, row)

  def good(v: list) -> bool:
    d = _predict(tbl, v)
    return d is not None and d <= thr

  if good(q):                          # already good: change nothing
    return _inputs(tbl, q)

  # Candidates: each good training build, with its differing free inputs
  # copied in. Fewest copied values come first.
  cands = []
  for g in tbl.rows:
    gd = _ydist(tbl, g)
    if gd is None or gd > thr: continue
    v, changed = list(q), 0
    for c in free:
      gv = g[c.at]
      if gv != "?" and gv != q[c.at]:
        v[c.at] = gv
        changed += 1
    cands.append((changed, gd, v))
  cands.sort(key=lambda t: (t[0], t[1]))

  best = None
  for _, _, v in cands:
    if good(v):
      best = v
      break

  if best is None:                     # nothing predicted good: do the best we can
    options = [q] + [v for _, _, v in cands]
    best = min(options, key=lambda v: (_predict(tbl, v)
                                       if _predict(tbl, v) is not None
                                       else float("inf")))
    return _inputs(tbl, best)

  # Greedy pruning: undo each change that is not needed for goodness.
  for c in free:
    if best[c.at] != q[c.at]:
      trial = list(best)
      trial[c.at] = q[c.at]
      if good(trial): best = trial

  return _inputs(tbl, best)

