# step5.py: which input columns are to blame for the bad rows?
"step5.py: which input columns are to blame for the bad rows?"
import bisect
import math
import ezr

BAD  = 0.75   # rows at or above this ydist quantile count as "bad"
BINS = 8      # equal-frequency bins for numeric inputs

def blame(train_csv: str) -> list[str]:
  "Input column names, most to blame for bad rows (bad on the goals) first."
  tbl = ezr.Tbl(ezr.csv(train_csv))

  def ydist_of(row):
    "How far ROW's goals are from ideal; infinite if any goal is unknown."
    if any(row[c.at] == "?" for c in tbl.y): return math.inf
    return ezr.ydist(tbl, row)

  known = [(y, r) for y, r in ((ydist_of(r), r) for r in tbl.rows)
           if y < math.inf]
  names = [c.txt for c in tbl.x]
  if len(known) < 2 or not tbl.x: return names

  sorted_ys = sorted(y for y, _ in known)
  cut   = sorted_ys[min(len(sorted_ys) - 1, int(BAD * len(sorted_ys)))]
  rows  = [r for _, r in known]
  labels = [y >= cut for y, _ in known]          # True = bad
  n  = len(rows)
  nb = sum(labels)
  ng = n - nb
  if nb == 0 or ng == 0: return names            # nothing to tell apart

  def bins_of(col):
    "One discrete bin per row for COL: its symbol, or its numeric quantile bin."
    if type(col) is ezr.Sym:
      return [r[col.at] for r in rows]
    xs = sorted(r[col.at] for r in rows if r[col.at] != "?")
    if not xs: return ["?"] * n
    edges = sorted({xs[i * len(xs) // BINS] for i in range(1, BINS)})
    return ["?" if r[col.at] == "?"
            else bisect.bisect_right(edges, r[col.at]) for r in rows]

  def info(bins):
    "Mutual information, in bits, between a column's bins and badness."
    cnt, bad = {}, {}
    for b, y in zip(bins, labels):
      cnt[b] = cnt.get(b, 0) + 1
      if y: bad[b] = bad.get(b, 0) + 1
    total = 0.0
    for b, c in cnt.items():
      bc = bad.get(b, 0)
      gc = c - bc
      for joint, pb in ((bc, nb), (gc, ng)):
        if joint: total += (joint / n) * math.log2((joint * n) / (c * pb))
    return total

  scored = [(info(bins_of(c)), c.txt) for c in tbl.x]
  return [name for _, name in sorted(scored, key=lambda t: -t[0])]

