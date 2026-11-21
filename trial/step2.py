# step2.py
import math
import ezr

_cache: dict[str, ezr.TBL] = {}   # load each training file once

def _load(train_csv: str) -> ezr.TBL:
  "Load TRAIN_CSV, and remember it for later calls."
  if train_csv not in _cache:
    _cache[train_csv] = ezr.Tbl(ezr.csv(train_csv))
  return _cache[train_csv]

def _pick(xs: list[float], p: float) -> float:
  "The value at fraction P of the sorted, non-empty list XS."
  return xs[min(len(xs)-1, max(0, round(p*(len(xs)-1))))]

def _query(tbl: ezr.TBL, row) -> ezr.ROW:
  "Align ROW to the table's columns. Missing cells (and goals) are '?'."
  q = ["?"] * len(tbl.names)
  if isinstance(row, dict):
    for at, name in enumerate(tbl.names):
      if name in row and row[name] is not None:
        v = row[name]
        q[at] = ezr.atom(v) if isinstance(v, str) and v != "?" else v
  else:  # a list, aligned with the names that are not goals
    keep = [at for at, n in enumerate(tbl.names) if n[-1] not in "+-"]
    for at, v in zip(keep, row):
      q[at] = ezr.atom(v) if isinstance(v, str) and v != "?" else v
  return q

def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
  "Per goal: (estimate, low, high), from the rows nearest to ROW."
  tbl   = _load(train_csv)
  goals = [c for c in tbl.y if type(c) is ezr.Num]
  if not tbl.rows:
    return {tbl.names[c.at]: (0.0, 0.0, 0.0) for c in goals}
  q    = _query(tbl, row)
  k    = min(len(tbl.rows), max(5, int(math.sqrt(len(tbl.rows)))))
  near = sorted(tbl.rows, key=lambda r: ezr.xdist(tbl, q, r))[:k]
  out  = {}
  for c in goals:
    vals = sorted(float(r[c.at]) for r in near if r[c.at] != "?")
    if not vals:  # no neighbour knows this goal: fall back to all rows
      vals = sorted(float(r[c.at]) for r in tbl.rows if r[c.at] != "?")
    if not vals:
      out[tbl.names[c.at]] = (0.0, 0.0, 0.0)
      continue
    est = sum(vals) / len(vals)
    lo  = min(_pick(vals, 0.10), est)   # the middle ~80% of neighbours
    hi  = max(_pick(vals, 0.90), est)
    out[tbl.names[c.at]] = (est, lo, hi)
  return out

