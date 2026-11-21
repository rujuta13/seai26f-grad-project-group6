# step5.py
"""Step 5: blame.  Which input columns go with the bad rows?

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


def _mean(xs: list[float]) -> float:
  "Mean of XS, or 0 if empty."
  return sum(xs) / len(xs) if xs else 0.0


def blame(train_csv: str) -> list[str]:
  """Input column names, most to blame for bad rows (bad on the goals) first.
  A row is bad if its ydist is in the worst quarter, good if in the best
  quarter.  Symbolic columns score the largest gap, over their values v,
  between how often v appears in bad rows and in good rows.  Numeric
  columns score the gap between their mean (normalised) in bad and good rows.
  Ties keep file order."""
  tbl  = _load(train_csv)
  rows = tbl.rows
  if not rows or not tbl.x: return [c.txt for c in tbl.x]

  ys   = [ezr.ydist(tbl, r) for r in rows]
  hi   = _quantile(ys, 0.75)
  lo   = _quantile(ys, 0.25)
  bad  = [r for r, y in zip(rows, ys) if y >= hi]
  good = [r for r, y in zip(rows, ys) if y <= lo]
  if not bad or not good: return [c.txt for c in tbl.x]

  def share(rs: ezr.ROWS, c, v) -> float:
    "Fraction of the known values of column C in RS that equal V."
    vals = [r[c.at] for r in rs if r[c.at] != "?"]
    return sum(1 for x in vals if x == v) / len(vals) if vals else 0.0

  def score(c) -> float:
    "How strongly column C separates bad rows from good rows."
    if isinstance(c, ezr.Num):
      vb = [ezr.norm(c, r[c.at]) for r in bad  if r[c.at] != "?"]
      vg = [ezr.norm(c, r[c.at]) for r in good if r[c.at] != "?"]
      return abs(_mean(vb) - _mean(vg))
    values = {r[c.at] for r in rows if r[c.at] != "?"}
    if not values: return 0.0
    return max(share(bad, c, v) - share(good, c, v) for v in values)

  scored = [(score(c), i, c.txt) for i, c in enumerate(tbl.x)]
  scored.sort(key=lambda t: (-t[0], t[1]))   # most blame first; ties by file order
  return [name for _, _, name in scored]

