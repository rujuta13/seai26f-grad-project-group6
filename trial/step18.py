# step18.py
"Step 18: fix, for the client. The fewest input changes that turn a bad player into a good one, leaving frozen inputs alone."
import heapq
import math
import os
import ezr

QUANTILE = 0.25   # a player is "good" if its goals are in the best 25% of train
K        = 20     # nearest training players consulted to predict goals
MAXVALS  = 16     # candidate values tried per input column

_CACHE: dict = {} # (path, mtime) -> (table, good cutoff, distance columns)

def _key(train_csv: str) -> tuple:
  "Identity of TRAIN_CSV: its absolute path and when it last changed."
  path = os.path.abspath(os.path.expanduser(train_csv))
  return (path, os.path.getmtime(path))

def _isnum(v) -> bool:
  "True for a usable finite number."
  return (isinstance(v, (int, float)) and not isinstance(v, bool)
          and math.isfinite(v))

def _name(s: str) -> str:
  "A column name without its +, - or ! suffix."
  return s[:-1] if s[-1:] in ("+", "-", "!") else s

def _coerce(v):
  "Strings become atoms; other values pass through."
  return ezr.atom(v) if isinstance(v, str) else v

def _full(tbl, row) -> list:
  """Lift ROW into a full-width row. Accepts a {column name: value} dict,
  a full-width list, or a list with goal columns removed (header order).
  Any cell not given is '?'. Non-numeric cells in numeric columns become '?'."""
  names = tbl.names
  out   = ["?"] * len(names)
  if isinstance(row, dict):
    for at, n in enumerate(names):
      if n in row:
        out[at] = _coerce(row[n])
      elif n[-1:] in ("+", "-", "!") and _name(n) in row:
        out[at] = _coerce(row[_name(n)])
  else:
    row = list(row)
    if len(row) == len(names):
      out = [_coerce(v) for v in row]
    else:
      keep = [i for i, n in enumerate(names) if n[-1:] not in ("+", "-", "!")]
      for i, v in zip(keep, row): out[i] = _coerce(v)
  for c in tbl.cols.values():
    if type(c) is ezr.Num and not _isnum(out[c.at]):
      out[c.at] = "?"
  return out

def _ydist(tbl, row) -> float | None:
  "Distance of ROW's numeric goals from their best; None if no goal is known."
  sq, n = 0.0, 0
  for c in tbl.y:
    if type(c) is not ezr.Num: continue
    v = row[c.at]
    if not _isnum(v): continue
    sq += (ezr.norm(c, v) - c.goal) ** 2
    n  += 1
  return (sq / n) ** .5 if n else None

def _good(d, thr) -> bool:
  "True if goal distance D is known and at or below the good cutoff THR."
  return d is not None and d <= thr

def _load(train_csv: str):
  "Train table, good cutoff (None if no known goals), and distance columns; cached per file version."
  key = _key(train_csv)
  if key not in _CACHE:
    tbl  = ezr.Tbl(ezr.csv(train_csv))
    ys   = sorted(d for d in (_ydist(tbl, r) for r in tbl.rows) if d is not None)
    thr  = ys[int(QUANTILE * (len(ys) - 1))] if ys else None
    cols = [c for c in tbl.x if type(c) is ezr.Num or len(c.has) < c.n]
    _CACHE[key] = (tbl, thr, cols)
  return _CACHE[key]

def _values(c, good_rows) -> list:
  "Candidate values for input column C: those seen in good train rows, thinned to MAXVALS."
  if type(c) is ezr.Sym:
    xs = sorted({r[c.at] for r in good_rows if r[c.at] != "?"}, key=str)
  else:
    xs = sorted({r[c.at] for r in good_rows if _isnum(r[c.at])})
  if len(xs) > MAXVALS:
    xs = [xs[(i * (len(xs) - 1)) // (MAXVALS - 1)] for i in range(MAXVALS)]
  return xs

def _gsq(tbl, memo: dict, c, val) -> list:
  "Squared gap between VAL and every train row's cell in column C (cached)."
  key = (c.at, val)
  if key not in memo:
    memo[key] = [ezr.gap(c, val, r[c.at]) ** 2 for r in tbl.rows]
  return memo[key]

def _d2(tbl, memo: dict, cols, v: list) -> list:
  "Squared input distance from full-width row V to every train row."
  total = [0.0] * len(tbl.rows)
  for c in cols:
    total = [t + g for t, g in zip(total, _gsq(tbl, memo, c, v[c.at]))]
  return total

def _predict(tbl, d2: list) -> float | None:
  """Predicted goal distance: the median goal value of the K train rows
  with the smallest squared input distances D2, then its goal distance."""
  idx   = heapq.nsmallest(K, range(len(d2)), key=d2.__getitem__)
  near  = [tbl.rows[i] for i in idx]
  guess = ["?"] * len(tbl.names)
  for c in tbl.y:
    if type(c) is not ezr.Num: continue
    xs = sorted(r[c.at] for r in near if _isnum(r[c.at]))
    if xs: guess[c.at] = xs[len(xs) // 2]
  return _ydist(tbl, guess)

def fix(train_csv, row, frozen=()) -> dict:
  """Inputs of bad ROW with as few values changed as possible so that it
  becomes good. ROW has its goal columns removed. Columns named in FROZEN
  never change. Returns {input column name: value}."""
  tbl, thr, cols = _load(train_csv)
  if thr is None:
    raise ValueError("fix: no training player has known goal values")
  if isinstance(frozen, str): frozen = (frozen,)
  frozen  = set(frozen)
  unknown = frozen - {c.txt for c in tbl.x}
  if unknown:
    raise ValueError(f"fix: frozen names are not input columns: {sorted(unknown)}")

  q         = _full(tbl, row)
  memo: dict = {}
  free      = [c for c in cols if c.txt not in frozen]
  good_rows = [r for r in tbl.rows if _good(_ydist(tbl, r), thr)]
  values    = {c.at: _values(c, good_rows) for c in free}

  v    = list(q)
  d2   = _d2(tbl, memo, cols, v)
  dist = _predict(tbl, d2)

  if not _good(dist, thr):
    # greedy forward: one value change at a time, stopping at the first good row
    for _ in range(len(free)):
      found = best = None
      for c in free:
        cur = v[c.at]
        old = _gsq(tbl, memo, c, cur)
        for val in values[c.at]:
          if val == cur: continue
          new = _gsq(tbl, memo, c, val)
          d2t = [t - a + b for t, a, b in zip(d2, old, new)]
          dt  = _predict(tbl, d2t)
          if dt is None: continue
          hit = (dt, c.at, val, d2t)
          if _good(dt, thr):
            if found is None or dt < found[0]: found = hit
          elif best is None or dt < best[0]:
            best = hit
      if found is not None:
        pick = found
      elif best is not None and (dist is None or best[0] < dist):
        pick = best
      else:
        break                           # no change helps: stop at the best found
      dist, d2 = pick[0], pick[3]
      v[pick[1]] = pick[2]
      if _good(dist, thr): break

  # greedy pruning: undo each change that is not needed for goodness
  if _good(dist, thr):
    for c in free:
      at = c.at
      if v[at] == q[at]: continue
      old = _gsq(tbl, memo, c, v[at])
      new = _gsq(tbl, memo, c, q[at])
      d2t = [t - a + b for t, a, b in zip(d2, old, new)]
      dt  = _predict(tbl, d2t)
      if _good(dt, thr):
        v[at], d2, dist = q[at], d2t, dt

  return {c.txt: v[c.at] for c in tbl.x}

