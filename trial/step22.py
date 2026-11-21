# step22.py: guess. Estimate each goal of a customer from his other numbers, with a range.
"step22.py: guess. Estimate each goal of a customer from his other numbers, with a range."
import functools
import math
import ezr

LO, HI = 0.10, 0.90   # the 10th..90th percentile: "fairly sure"
K_MIN  = 5            # fewest neighbours we will trust

@functools.lru_cache(maxsize=8)
def _load(train_csv: str):
  "Read TRAIN_CSV once; guess() is called once per customer."
  return ezr.Tbl(ezr.csv(train_csv))

def _val(v):
  "Cells from dicts may still be strings: coerce them the way csv does."
  return ezr.atom(v) if isinstance(v, str) else v

def _align(tbl, row):
  """ROW in TBL's column order. A dict may be keyed by full column name or by
  name without its goal suffix; missing cells are '?'."""
  if isinstance(row, dict):
    out = []
    for n in tbl.names:
      if n in row:
        out.append(_val(row[n]))
      elif n[-1] in "+-!" and n[:-1] in row:
        out.append(_val(row[n[:-1]]))
      else:
        out.append("?")
    return out
  return [_val(v) for v in row]

def _pct(xs: list, q: float) -> float:
  "The q-th quantile of sorted XS, by linear interpolation."
  i      = q * (len(xs) - 1)
  lo, hi = math.floor(i), math.ceil(i)
  return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)

def _nums(rows, at) -> list[float]:
  "Sorted floats of column AT over ROWS, skipping unknown and non-numeric cells."
  out = []
  for r in rows:
    v = r[at]
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
      out.append(float(v))
  return sorted(out)

def guess(train_csv: str, row) -> dict[str, tuple[float, float, float]]:
  """For each goal of TRAIN_CSV: (estimate, low, high). The estimate is the median
  goal value of the nearest training rows (by their inputs); low..high is the 10th..90th
  percentile of those values. ROW has its goal columns removed; only its inputs are used."""
  tbl = _load(train_csv)
  if not tbl.y: raise ValueError("no goal columns (names ending in + or -)")
  rows = tbl.rows
  if not rows: raise ValueError("no training rows to guess from")
  q = _align(tbl, row)
  for g in tbl.y: q[g.at] = "?"            # this customer's goals are unknown
  near = sorted(rows, key=lambda r: ezr.xdist(tbl, q, r))
  k    = min(len(rows), max(K_MIN, int(math.sqrt(len(rows)))))
  out  = {}
  for g in tbl.y:
    vs = _nums(near[:k], g.at)
    if not vs:   # no close neighbour has this goal: use the whole column
      vs = _nums(rows, g.at)
    if not vs:   # the column is empty: fall back on its summary
      mu, sd = float(g.mu), float(g.sd)
      out[g.txt] = (mu, mu - sd, mu + sd)
    else:
      out[g.txt] = (_pct(vs, 0.5), _pct(vs, LO), _pct(vs, HI))
  return out

