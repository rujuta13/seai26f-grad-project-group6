#!/usr/bin/env python3 -B
"step2.py: guess each goal of an unbenchmarked flag set, with a range."
from functools import lru_cache
from math import ceil, floor, nan
from ezr import Tbl, csv, xdist

K  = 10     # nearest training rows that vote on each guess
LO = 0.10   # low end of the range we are fairly sure of
HI = 0.90   # high end of that range

@lru_cache(maxsize=4)
def _load(train_csv: str):
  "The training table, read once per file."
  return Tbl(csv(train_csv))

def _quantile(xs: list, p: float) -> float:
  "The P-th quantile of sorted XS, by linear interpolation."
  pos = p * (len(xs) - 1)
  lo, hi = floor(pos), ceil(pos)
  return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)

def _full(tbl, row) -> list:
  """Expand ROW to a full table row, with goals as '?'.  ROW is either
  a dict {column name: value} or a list in header order minus goals."""
  names = tbl.names
  if isinstance(row, dict):
    got = row
  else:
    free = [s for s in names if s[-1] not in "+-"]
    got  = dict(zip(free, row))
  return [got.get(s, "?") for s in names]

def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
  "For each goal: (median, low, high) over the K nearest training rows."
  tbl = _load(str(train_csv))
  if not tbl.rows: raise ValueError(f"no rows in {train_csv}")
  me   = _full(tbl, row)
  near = sorted(tbl.rows, key=lambda r: xdist(tbl, me, r))[:K]
  out  = {}
  for c in tbl.y:
    if c.txt[-1] not in "+-": continue      # goals only
    xs = sorted(r[c.at] for r in near if r[c.at] != "?")
    if not xs:
      out[c.txt] = (nan, nan, nan)
      continue
    out[c.txt] = (_quantile(xs, .5), _quantile(xs, LO), _quantile(xs, HI))
  return out

