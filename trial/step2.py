# step2.py: guess a flag set's goal values, and how sure we are.
"step2.py: guess a flag set's goal values, and how sure we are."
import functools
import math
import ezr

LO, HI = 0.10, 0.90   # the 10th..90th percentile: "fairly sure"
K_MIN  = 5            # fewest neighbours we will trust

@functools.lru_cache(maxsize=8)
def _load(train_csv: str):
  "Read TRAIN_CSV once; guess() is called once per flag set."
  return ezr.Tbl(ezr.csv(train_csv))

def _pct(xs: list, q: float) -> float:
  "The q-th quantile of sorted XS, by linear interpolation."
  i      = q * (len(xs) - 1)
  lo, hi = math.floor(i), math.ceil(i)
  return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)

def guess(train_csv: str, row: dict) -> dict[str, tuple[float, float, float]]:
  "For each goal: (median, 10th pct, 90th pct) of the nearest training rows."
  tbl  = _load(train_csv)
  rows = tbl.rows
  q    = [row.get(name, "?") for name in tbl.names]  # align to columns
  near = sorted(rows, key=lambda r: ezr.xdist(tbl, q, r))
  k    = min(len(rows), max(K_MIN, int(math.sqrt(len(rows)))))
  out  = {}
  for g in tbl.y:
    vs = sorted(float(r[g.at]) for r in near[:k] if r[g.at] != "?")
    if not vs:   # no close neighbour has this goal: use the whole column
      vs = sorted(float(r[g.at]) for r in rows if r[g.at] != "?")
    if not vs:   # the column is empty: fall back on its summary
      out[g.txt] = (float(g.mu), float(g.mu - g.sd), float(g.mu + g.sd))
    else:
      out[g.txt] = (_pct(vs, 0.5), _pct(vs, LO), _pct(vs, HI))
  return out

