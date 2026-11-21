# step5.py
#!/usr/bin/env python3 -B
"""
step5.py: blame.  Which input columns go with the bad rows?
A row is bad if its ydist (all goals together, so a fast build that still
eats CPU counts as bad) is in the worst quarter of the training rows.
Each input column is scored by how much of the bad/good split its values
explain (eta-squared on the 0/1 bad flag): 0 = no blame, 1 = the column
alone separates bad rows from good ones.  Symbols group by value; numbers
group by quartile.  Reuses ezr for distance and step2 for the cached table.
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

  def eta(col):
    "Share of the bad-flag variance that COL's groups explain."
    idx = [i for i, r in enumerate(rows) if r[col.at] != "?"]
    if len(idx) < 2: return 0.0
    if type(col) is ezr.Sym:
      key = {i: rows[i][col.at] for i in idx}
    else:
      order = sorted(idx, key=lambda i: rows[i][col.at])
      size  = max(1, len(order)//4)
      key   = {i: min(3, j//size) for j, i in enumerate(order)}
    grand = sum(bad[i] for i in idx)/len(idx)
    total = sum((bad[i]-grand)**2 for i in idx)
    if total <= 0: return 0.0
    groups = {}
    for i in idx: groups.setdefault(key[i], []).append(bad[i])
    between = sum(len(g)*(sum(g)/len(g)-grand)**2 for g in groups.values())
    return between/total

  scored = [(-eta(c), c.txt) for c in tbl.x]
  return [name for _, name in sorted(scored)]

