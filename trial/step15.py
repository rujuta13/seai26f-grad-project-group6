# step15.py
#!/usr/bin/env python3 -B
"""
step15.py: blame.  Which input columns go with the bad rows?
A row is bad if its ydist (all goals together) is in the worst quarter of
the training rows with every goal known.  Each input column is scored by
how much of the bad/good split its values explain: eta-squared on the 0/1
bad flag, adjusted for the number of groups.  Symbols group by value;
numbers group by quartile.

Why this is not step5.blame unchanged: plain eta-squared rises toward 1 for
any column with many distinct values, because each group then holds one or
two rows and explains the flag by chance.  The adjustment
1 - (1-eta)(n-1)/(n-k) removes that chance share, so identifier-like
columns score near 0.  Reuses ezr for distance and step2 for the cached table.
"""
import ezr
import step2

def blame(train_csv: str) -> list[str]:
  "Input column names, most to blame for bad rows (bad on the goals) first."
  tbl  = step2._tbl(train_csv)
  rows = [r for r in tbl.rows if all(r[c.at] != "?" for c in tbl.y)]
  if not rows: return [c.txt for c in tbl.x]
  ys   = [ezr.ydist(tbl, r) for r in rows]
  cut  = sorted(ys)[min(len(ys)-1, int(0.75*len(ys)))]
  bad  = [1 if y >= cut else 0 for y in ys]

  def blame_of(col) -> float:
    "Adjusted share of the bad-flag variance that COL's groups explain."
    idx = [i for i, r in enumerate(rows) if r[col.at] != "?"]
    n   = len(idx)
    if n < 2: return 0.0
    if type(col) is ezr.Sym:
      key = {i: rows[i][col.at] for i in idx}
    else:
      order = sorted(idx, key=lambda i: rows[i][col.at])
      size  = max(1, n//4)
      key   = {i: min(3, j//size) for j, i in enumerate(order)}
    grand = sum(bad[i] for i in idx)/n
    total = sum((bad[i]-grand)**2 for i in idx)
    if total <= 0: return 0.0
    groups = {}
    for i in idx: groups.setdefault(key[i], []).append(bad[i])
    k = len(groups)
    if n <= k: return 0.0          # every row its own group: no evidence
    between = sum(len(g)*(sum(g)/len(g)-grand)**2 for g in groups.values())
    eta = between/total
    return max(0.0, 1 - (1-eta)*(n-1)/(n-k))

  scored = [(-blame_of(c), c.txt) for c in tbl.x]
  return [name for _, name in sorted(scored)]

