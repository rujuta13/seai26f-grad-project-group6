# step2.py
"""Step 2: guess each goal's value for a flag set, with a range.

Uses only the standard library plus ezr.
"""
import math
import ezr


def _load(train_csv: str) -> ezr.TBL:
  "Read TRAIN_CSV into an ezr table, keeping file order."
  return ezr.Tbl(ezr.csv(train_csv))


def _quantile(xs: list[float], p: float) -> float:
  "The p-th quantile of XS, linearly interpolated.  XS must be non-empty."
  xs     = sorted(xs)
  i      = p * (len(xs) - 1)
  lo, hi = math.floor(i), math.ceil(i)
  return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def _query(tbl: ezr.TBL, row) -> ezr.ROW:
  "Lay the flag values of ROW into a full-width ezr row; missing cells are '?'."
  q = ["?"] * len(tbl.names)
  if isinstance(row, dict):              # {column name: value}
    for c in tbl.x:
      if c.txt in row: q[c.at] = row[c.txt]
  else:                                  # positional, in x-column order
    for c, v in zip(tbl.x, row): q[c.at] = v
  return q


def guess(train_csv: str, row) -> dict[str, tuple[float, float, float]]:
  """For each goal: (estimate, low, high).  The estimate is the median
  goal value of the training rows whose flags are nearest to ROW; low and
  high are the 10th and 90th percentiles of those same values."""
  tbl   = _load(train_csv)
  rows  = tbl.rows
  goals = list(tbl.y)
  if not goals: return {}
  if not rows:  return {c.txt: (0.0, 0.0, 0.0) for c in goals}
  q    = _query(tbl, row)
  k    = min(len(rows), max(5, int(math.sqrt(len(rows)))))
  near = sorted(rows, key=lambda r: ezr.xdist(tbl, q, r))[:k]
  out  = {}
  for c in goals:
    vs = [r[c.at] for r in near if r[c.at] != "?"]
    if not vs:                           # no nearby evidence: use all rows
      vs = [r[c.at] for r in rows if r[c.at] != "?"]
    if not vs:
      out[c.txt] = (0.0, 0.0, 0.0)
      continue
    est = _quantile(vs, 0.5)
    lo  = min(est, _quantile(vs, 0.1))
    hi  = max(est, _quantile(vs, 0.9))
    out[c.txt] = (est, lo, hi)
  return out

