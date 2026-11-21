# step25.py
import ezr

BAD = 0.75  # rows in the worst quarter of goal score count as bad

def blame(train_csv) -> list[str]:
  "Input column names, most to blame for bad rows (bad on the goals) first."
  tbl   = ezr.Tbl(ezr.csv(train_csv))
  names = [tbl.names[c.at] for c in tbl.x]
  goals = [c for c in tbl.y if type(c) is ezr.Num]
  if not goals or not tbl.x: return sorted(names)
  # rows whose every goal is a number can be ranked; the rest are skipped
  rows  = [r for r in tbl.rows
           if all(isinstance(r[c.at], (int, float))
                  and not isinstance(r[c.at], bool) for c in goals)]
  if not rows: return sorted(names)
  ys    = [ezr.dist((abs(ezr.norm(c, r[c.at]) - c.goal) for c in goals),
                    len(goals))
           for r in rows]
  cut   = sorted(ys)[min(len(ys) - 1, int(BAD * (len(ys) - 1)))]
  bad   = [y >= cut for y in ys]
  total = sum(bad)
  base  = total / len(rows)
  scores: dict[str, float] = {}
  for col in tbl.x:
    name, best = tbl.names[col.at], 0.0
    for _, go in ezr.candidates(col, rows):
      yes     = [i for i, r in enumerate(rows) if go(r)]
      n_yes   = len(yes)
      n_no    = len(rows) - n_yes
      if n_yes < ezr.the.Leaf or n_no < ezr.the.Leaf: continue
      bad_yes = sum(bad[i] for i in yes)
      ry      = bad_yes / n_yes                 # bad share where the split holds
      rn      = (total - bad_yes) / n_no        # bad share where it does not
      best    = max(best, ry - base, rn - base)
    scores[name] = best
  return sorted(names, key=lambda n: (-scores[n], n))

