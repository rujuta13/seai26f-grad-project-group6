# step2.py
from math import sqrt
import ezr

_cache = {}   # train_csv -> TBL, loaded once

def _table(train_csv: str):
  "Load TRAIN_CSV once and reuse it on later calls."
  if train_csv not in _cache:
    _cache[train_csv] = ezr.Tbl(ezr.csv(train_csv))
  return _cache[train_csv]

def _atom(v):
  "Coerce strings with ezr's rules; leave numbers alone."
  return ezr.atom(v) if isinstance(v, str) else v

def _row(tbl, row) -> ezr.ROW:
  """Put ROW (a dict or list, with goal cells removed) into the full
  column layout of TBL. Missing cells, including goals, become '?'."""
  goals = {c.at for c in tbl.y}
  full  = ["?"] * len(tbl.names)
  if isinstance(row, dict):
    for at, name in enumerate(tbl.names):
      if name in row: full[at] = _atom(row[name])
  else:
    if len(row) == len(tbl.names):
      slots = list(range(len(tbl.names)))
    else:   # goals removed: the cells are the non-goal columns, in order
      slots = [at for at in range(len(tbl.names)) if at not in goals]
    for at, v in zip(slots, row): full[at] = _atom(v)
  return full

def _q(vs: list, p: float):
  "The p-th quantile of sorted VS, by index."
  return vs[min(len(vs) - 1, int(p * len(vs)))]

def guess(train_csv: str, row) -> dict[str, tuple[float, float, float]]:
  """For each goal: (estimate, low, high). The estimate is the mean goal
  value of the rows nearest to ROW (over the x columns). The range is
  the 10th to 90th percentile of those neighbours, widened to include
  the estimate."""
  tbl  = _table(train_csv)
  rows = tbl.rows
  if not rows: raise ValueError(f"no rows in {train_csv}")
  q    = _row(tbl, row)
  k    = min(len(rows), max(3, int(sqrt(len(rows)))))
  near = sorted(rows, key=lambda r: ezr.xdist(tbl, r, q))[:k]
  out  = {}
  for c in tbl.y:
    vs = sorted(r[c.at] for r in near if r[c.at] != "?")
    if not vs:   # no neighbour has this goal: fall back to the whole table
      vs = sorted(r[c.at] for r in rows if r[c.at] != "?")
    est = sum(vs) / len(vs)
    lo  = min(_q(vs, 0.1), est)
    hi  = max(_q(vs, 0.9), est)
    out[c.txt] = (est, lo, hi)
  return out

