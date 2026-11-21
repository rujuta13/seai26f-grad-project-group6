# step12.py
#!/usr/bin/env python3 -B
"Guess each goal of a build from its other inputs, with a range we are fairly sure of, via ezr."
import math
import ezr

_cache: dict[str, tuple] = {}   # train_csv -> (input columns, goals as (name, at), rows)

def _cell(v) -> ezr.ATOM:
  "V as one cell: '?' if blank, unknown, None, or not finite; otherwise its atom."
  if v is None: return "?"
  if isinstance(v, str): v = ezr.atom(v.strip() or "?")
  if isinstance(v, float) and not math.isfinite(v): return "?"
  return v

def _num(v) -> bool:
  "True for a usable number: an int or float, but not a bool."
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def _fit(col, v) -> ezr.ATOM:
  "V as COL expects it: a number for a Num (anything else is unknown), any atom for a Sym."
  v = _cell(v)
  if type(col) is ezr.Num and not _num(v): return "?"
  return v

def _pct(vs: list, p: float) -> float:
  "The value at fraction P of the sorted VS (nearest rank)."
  vs = sorted(vs)
  return vs[round(p * (len(vs) - 1))]

def _load(train_csv: str) -> tuple:
  "Read TRAIN_CSV once; return its input columns, its goals as (name, at), and its rows."
  if train_csv not in _cache:
    src   = ezr.csv(train_csv)
    head  = [str(n) for n in (next(src, None) or [])]
    width = len(head)
    rows  = []
    for r in src:
      r = [_cell(v) for v in r]
      rows.append((r + ["?"] * width)[:width])
    n     = len(rows)
    cols, goals = [], []
    for at, name in enumerate(head):
      vals = [r[at] for r in rows if r[at] != "?"]
      if name[-1:] in ("+", "-"):              # a goal: keep it out of the inputs
        if any(_num(v) for v in vals): goals.append((name, at))
        continue
      if name == "num" or name[-1:] == "X" or not vals: continue
      numeric = all(_num(v) for v in vals)
      if not numeric and len(set(vals)) > max(1, n // 2):
        continue                               # an identifier, like a name: no signal
      col = ezr.Num(name, at) if numeric else ezr.Sym(name, at)
      for v in vals: ezr.add(col, v)
      cols.append(col)
    _cache[train_csv] = (cols, goals, rows)
  return _cache[train_csv]

def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
  """Estimate each goal of ROW (a dict with its goal columns removed).
  Returns {goal: (estimate, low, high)}. The estimate is the median goal value
  of the nearest labelled training rows over the input columns; low..high is
  the 10th..90th percentile of those same neighbours, the range we are fairly
  sure the true value falls in. A wide range means the neighbours disagree."""
  cols, goals, rows = _load(train_csv)
  if not goals: return {}
  q     = {c.at: _fit(c, row.get(c.txt, "?")) for c in cols}
  width = max(1, len(cols))
  def dist(r: list) -> float:
    "Distance from the query row to training row R, over the input columns."
    return ezr.dist((ezr.gap(c, q[c.at], r[c.at]) for c in cols), width)
  k    = max(3, min(len(rows), int(len(rows) ** 0.5)))
  near = sorted(rows, key=dist)[:k]
  out  = {}
  for name, at in goals:
    vs = [float(r[at]) for r in near if _num(r[at])]
    if not vs:                       # no neighbour knows this goal: use all rows
      vs = [float(r[at]) for r in rows if _num(r[at])]
    est, low, high = _pct(vs, 0.5), _pct(vs, 0.1), _pct(vs, 0.9)
    out[name] = (est, low, high)
  return out

