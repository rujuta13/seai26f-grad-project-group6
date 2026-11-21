# step5.py
import ezr

BAD = 0.75  # rows in the worst quarter of ydist count as bad

def blame(train_csv) -> list[str]:
  "Input column names, most to blame for bad rows (bad on the goals) first."
  tbl  = ezr.Tbl(ezr.csv(train_csv))
  rows = tbl.rows
  names = [tbl.names[c.at] for c in tbl.x]
  if not rows or not tbl.x: return names
  ys   = [ezr.ydist(tbl, r) for r in rows]
  cut  = sorted(ys)[min(len(ys) - 1, int(BAD * (len(ys) - 1)))]
  bad  = [y >= cut for y in ys]
  base = sum(bad) / len(rows)
  scores: dict[str, float] = {}
  for col in tbl.x:
    name, best = tbl.names[col.at], 0.0
    for _, go in ezr.candidates(col, rows):
      yes = [i for i, r in enumerate(rows) if go(r)]
      no  = [i for i, r in enumerate(rows) if not go(r)]
      if len(yes) < ezr.the.Leaf or len(no) < ezr.the.Leaf: continue
      ry = sum(bad[i] for i in yes) / len(yes)
      rn = sum(bad[i] for i in no) / len(no)
      best = max(best, ry - base, rn - base)
    scores[name] = best
  return sorted(names, key=lambda n: (-scores[n], n))

