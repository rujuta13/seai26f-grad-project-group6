# step25.py: blame. Which input columns are most to blame for the bad rows?
"step25.py: blame. Which input columns are most to blame for the bad rows?"
import bisect
import math
import ezr

BAD  = 0.75   # rows at or above this ydist quantile count as "bad"
BINS = 8      # equal-frequency bins for numeric inputs

def _ydist_or_inf(tbl, row):
  "How far ROW's goals are from ideal; infinite if there are no goals or any goal is unknown."
  if not tbl.y or any(row[c.at] == "?" for c in tbl.y): return math.inf
  return ezr.ydist(tbl, row)

def _bins(col, rows):
  "One discrete bin per row for COL: its symbol, or its numeric quantile bin; '?' if unknown."
  if type(col) is ezr.Sym:
    return [r[col.at] for r in rows]
  xs = sorted(r[col.at] for r in rows if r[col.at] != "?")
  if not xs: return ["?"] * len(rows)
  edges = sorted({xs[i * len(xs) // BINS] for i in range(1, BINS)})
  return ["?" if r[col.at] == "?" else bisect.bisect_right(edges, r[col.at])
          for r in rows]

def _info(bins, labels):
  "Mutual information, in bits, between BINS and the bad LABELS (True = bad)."
  n  = len(labels)
  nb = sum(labels)
  ng = n - nb
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

def blame(train_csv) -> list[str]:
  """Input column names, most to blame for bad rows (bad on the goals) first.
  Bad means goal distance in the worst BAD fraction of rows with all goals known;
  goals include RESPONSE+ (never responds) and COMPLAIN- (complains)."""
  tbl   = ezr.Tbl(ezr.csv(train_csv))
  names = [c.txt for c in tbl.x]
  known = [(y, r) for y, r in ((_ydist_or_inf(tbl, r), r) for r in tbl.rows)
           if y < math.inf]
  if len(known) < 2 or not tbl.x: return names
  ys     = sorted(y for y, _ in known)
  cut    = ys[min(len(ys) - 1, int(BAD * len(ys)))]
  rows   = [r for _, r in known]
  labels = [y >= cut for y, _ in known]          # True = bad
  nb     = sum(labels)
  if nb == 0 or nb == len(labels): return names  # nothing to tell apart
  scored = [(_info(_bins(c, rows), labels), c.txt) for c in tbl.x]
  return [name for _, name in sorted(scored, key=lambda t: -t[0])]

