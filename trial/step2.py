# step2.py
#!/usr/bin/env python3 -B
"""
step2.py: guess.  Given a flag set, guess each goal, with a range.
Reuses ezr for tables and distance.  The training rows nearest the
flag set (by input columns) vote: estimate = median, range = the
10th..90th percentile of their goal values.
"""
from functools import lru_cache
from math import sqrt
import ezr

@lru_cache(maxsize=None)
def _tbl(train_csv: str):
  "Load the training table once per file."
  return ezr.Tbl(ezr.csv(train_csv))

def _quantile(xs: list, q: float) -> float:
  "Linear interpolation between ranks.  XS must be sorted and non-empty."
  i  = q * (len(xs) - 1)
  lo = int(i)
  hi = min(lo + 1, len(xs) - 1)
  return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)

def _probe(tbl, row: dict) -> list:
  "Turn a dict row (goals removed) into an ezr row; missing cells are '?'."
  out = []
  for name in tbl.names:
    v = row.get(name, "?")
    out.append("?" if v is None else v)
  return out

def guess(train_csv: str, row: dict) -> dict[str, tuple[float, float, float]]:
  """Return {goal: (estimate, low, high)} for every goal column.
  Goal keys are the header names (e.g. 'Performance-').  ESTIMATE is the
  median goal value of the nearest training rows; LOW..HIGH is their
  10th..90th percentile range."""
  tbl   = _tbl(train_csv)
  probe = _probe(tbl, row)
  k     = max(3, int(sqrt(len(tbl.rows))))
  near  = sorted(tbl.rows,
                 key=lambda r: ezr.xdist(tbl, probe, r))[:k]
  out = {}
  for col in tbl.y:
    xs = sorted(r[col.at] for r in near if r[col.at] != "?")
    if not xs:   # no neighbour knows this goal: fall back to every row
      xs = sorted(r[col.at] for r in tbl.rows if r[col.at] != "?")
    out[col.txt] = (_quantile(xs, .5),
                    _quantile(xs, .1),
                    _quantile(xs, .9))
  return out

