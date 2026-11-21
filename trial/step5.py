# step5.py
"""
step5: blame.  Which input columns go with the bad rows?  A row is bad
when its goals are far from ideal (high ydist).  For each input column,
find the single value, or the single side of a single cut, where bad
rows are most concentrated.  Score it by lift: the bad-row rate on that
side divided by the bad-row rate overall.  Reuses ezr's table,
candidates, and ydist.
"""
import ezr

BAD_FRAC = 0.25   # the worst 25% of training rows count as bad


def _lift(bad_in: int, n_in: int, base: float) -> float:
  "How much denser bad rows are on one side of a split than overall."
  if n_in == 0 or base == 0: return 0.0
  return (bad_in / n_in) / base


def blame(train_csv: str) -> list[str]:
  """Input column names, most to blame for bad rows (bad on the goals)
  first. Ties keep the order of the header."""
  tbl  = ezr.Tbl(ezr.csv(train_csv))
  rows = tbl.rows
  if not rows:
    return [tbl.names[c.at] for c in tbl.x]
  ys    = [ezr.ydist(tbl, r) for r in rows]     # higher = worse
  order = sorted(ys)
  cut   = order[min(len(order) - 1, int((1 - BAD_FRAC) * len(order)))]
  bad   = [y >= cut for y in ys]
  nbad  = sum(bad)
  base  = nbad / len(rows)                      # > 0: the worst row is bad
  need  = max(ezr.the.Leaf, len(rows) // 10)    # ignore tiny sides
  score = {}
  for col in tbl.x:
    best = 0.0
    for _, go in ezr.candidates(col, rows):
      hit  = [bool(go(r)) for r in rows]
      nyes = sum(hit)
      nno  = len(rows) - nyes
      byes = sum(b and h for b, h in zip(bad, hit))
      bno  = nbad - byes
      if nyes >= need: best = max(best, _lift(byes, nyes, base))
      if nno  >= need: best = max(best, _lift(bno,  nno,  base))
    score[col.at] = best
  ranked = sorted(tbl.x, key=lambda c: -score[c.at])
  return [tbl.names[c.at] for c in ranked]

