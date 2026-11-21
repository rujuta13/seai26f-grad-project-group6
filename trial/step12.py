# step12.py
"""Step 12: guess.  Before a scout is sent, guess each goal of a player from
his other numbers, with a range calibrated on the training data.

The estimate is the median goal value of the training rows nearest to the
player.  The range is that estimate widened by the 90th percentile of how
far the same kind of guess missed on training rows that were held out of
their own neighbourhood, so the range is only as narrow as the data allows.

Uses only the standard library plus ezr.
"""
import math
import random
import ezr

_CAL   = 150     # train rows used to calibrate the range
_COVER = 0.9     # the range should hold about this share of the time


def _load(train_csv: str) -> ezr.TBL:
  "Read TRAIN_CSV into an ezr table, keeping file order."
  return ezr.Tbl(ezr.csv(train_csv))


def _quantile(xs: list[float], p: float) -> float:
  "The p-th quantile of XS, linearly interpolated.  XS must be non-empty."
  xs     = sorted(xs)
  i      = p * (len(xs) - 1)
  lo, hi = math.floor(i), math.ceil(i)
  return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def _full(tbl: ezr.TBL, row) -> ezr.ROW:
  "Lay ROW out full-width, with '?' for anything missing; strings are cast."
  q = ["?"] * len(tbl.names)
  if isinstance(row, dict):                    # {column name: value}
    for c in tbl.cols.values():
      name = tbl.names[c.at]
      if name in row: q[c.at] = row[name]
  elif len(row) == len(tbl.names):             # already full-width
    q = list(row)
  else:                                        # positional, x-column order
    for c, v in zip(tbl.x, row): q[c.at] = v
  return [ezr.atom(v) if isinstance(v, str) else v for v in q]


def _enc(c: ezr.COL, v):
  "An input cell for distance: None if unknown, symbols as-is, numbers scaled to 0..1."
  if isinstance(v, str) and v == "?": return None
  if isinstance(c, ezr.Num):
    if not isinstance(v, (int, float)): return None
    return ezr.norm(c, v)
  return v


def _goal(v) -> float | None:
  "A goal cell as a float, or None if it is unknown."
  if isinstance(v, bool) or not isinstance(v, (int, float)): return None
  return float(v)


def _dist(eq: list, e: list, sym: list[bool]) -> float:
  "Root mean square gap between two encoded input rows, as ezr.xdist."
  s = 0.0
  for j, (a, b) in enumerate(zip(eq, e)):
    if a is None or b is None: g = 1.0
    elif sym[j]:               g = 0.0 if a == b else 1.0
    else:                      g = abs(a - b)
    s += g * g
  return math.sqrt(s / len(eq)) if eq else 0.0


def guess(train_csv: str, row) -> dict[str, tuple[float, float, float]]:
  """For each goal: (estimate, low, high).  ROW has its goal columns removed.
  The estimate is the median goal value of the training rows nearest to ROW.
  LOW and HIGH bracket it by the calibrated miss, clamped to the goal values
  seen in training, so the range is never wider than the data allows."""
  tbl   = _load(train_csv)
  rows  = tbl.rows
  goals = list(tbl.y)
  if not goals: return {}
  if not rows:  return {c.txt: (0.0, 0.0, 0.0) for c in goals}

  n   = len(rows)
  k   = min(n, max(5, int(math.sqrt(n))))
  sym = [not isinstance(c, ezr.Num) for c in tbl.x]
  E   = [[_enc(c, r[c.at]) for c in tbl.x] for r in rows]
  eq  = [_enc(c, _full(tbl, row)[c.at]) for c in tbl.x]

  def near(e: list, skip: int | None = None) -> list[int]:
    "Indices of the K training rows nearest encoded row E, leaving out SKIP."
    if not tbl.x:
      order = [i for i in range(n) if i != skip]
    else:
      order = sorted((i for i in range(n) if i != skip),
                     key=lambda i: (_dist(e, E[i], sym), i))
    return order[:k]

  qn   = near(eq)
  rng  = random.Random(ezr.the.Seed)           # local rng: no global side effects
  cal  = rng.sample(range(n), min(_CAL, n))
  cnear = {i: near(E[i], skip=i) for i in cal} # held out of their own neighbourhood

  out = {}
  for c in goals:
    gv = {}
    for i, r in enumerate(rows):
      v = _goal(r[c.at])
      if v is not None: gv[i] = v
    if not gv:
      out[c.txt] = (0.0, 0.0, 0.0)
      continue
    vals_all = list(gv.values())

    def estimate(idxs: list[int]) -> float:
      "Median goal value of the known goals among IDXS (all rows if none)."
      vs = [gv[i] for i in idxs if i in gv]
      if not vs: vs = vals_all
      return _quantile(vs, 0.5)

    est   = estimate(qn)
    resid = [abs(gv[i] - estimate(cnear[i])) for i in cal if i in gv]
    half  = (_quantile(resid, _COVER) if resid
             else (max(vals_all) - min(vals_all)) / 2)
    lo    = max(min(vals_all), est - half)
    hi    = min(max(vals_all), est + half)
    out[c.txt] = (float(est), float(min(lo, est)), float(max(hi, est)))
  return out

