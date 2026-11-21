# step5.py
#!/usr/bin/env python3 -B
"Which input columns are to blame for bad rows (bad on the goals), via ezr."
import ezr

def blame(train_csv) -> list[str]:
  """Input column names, most to blame for bad rows (bad on the goals) first.
  A row is bad if its goals are in the worst quarter by goal distance (ydist).
  Each input column is scored by its largest lift: how much more often bad rows
  fall on one side of a candidate split than bad rows fall in the table overall.
  Rows with an unknown goal cell are ignored. Ties keep file order."""
  tbl   = ezr.Tbl(ezr.csv(train_csv))
  goals = [c for c in tbl.y if c.txt[-1] in "+-"]
  xs    = tbl.x
  if not goals or not xs: return []

  def gydist(row) -> float | None:
    "Distance of ROW's goals from the ideal, or None if a goal cell is unknown."
    if any(row[c.at] == "?" for c in goals): return None
    return ezr.dist((abs(ezr.norm(c, row[c.at]) - c.goal) for c in goals),
                    len(goals))

  rows, ys = [], []
  for r in tbl.rows:
    y = gydist(r)
    if y is not None: rows.append(r); ys.append(y)
  n = len(rows)
  if n == 0: return [c.txt for c in xs]

  k    = max(1, n // 4)                          # the worst quarter
  order = sorted(range(n), key=lambda i: ys[i])
  bad   = set(order[-k:])                        # highest ydist = worst
  base  = len(bad) / n                           # bad rate overall
  leaf  = max(1, ezr.the.Leaf)                   # min rows on a side

  def score(col) -> float:
    "Largest lift in bad rate of any side of any candidate split of COL."
    best = -1.0                                  # lift is always >= -1
    for _, go in ezr.candidates(col, rows):
      yes  = [i for i in range(n) if go(rows[i])]
      seen = set(yes)
      no   = [i for i in range(n) if i not in seen]
      for side in (yes, no):
        if len(side) >= leaf:
          rate = sum(i in bad for i in side) / len(side)
          best = max(best, rate - base)
    return best

  scored = [(score(c), c) for c in xs]
  scored.sort(key=lambda p: -p[0])              # stable: ties keep file order
  return [c.txt for _, c in scored]

