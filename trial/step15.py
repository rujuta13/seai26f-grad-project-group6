# step15.py
import ezr

BAD = 0.75  # rows in the worst quarter of ydist count as bad

def _is_num(v) -> bool:
  "True if V is a real number (a bool is not)."
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def _goals(tbl: ezr.TBL) -> list:
  "The numeric goal columns of TBL (names ending + or -)."
  return [c for c in tbl.y if type(c) is ezr.Num]

def _complete(goals: list, r: ezr.ROW) -> bool:
  "True if every goal cell of R is a number."
  return all(_is_num(r[c.at]) for c in goals)

def _ydist(goals: list, r: ezr.ROW) -> float:
  "How far R's goals are from the best they could be (0 is best, 1 is worst)."
  return ezr.dist((abs(ezr.norm(c, r[c.at]) - c.goal) for c in goals),
                  len(goals))

def blame(train_csv) -> list[str]:
  "Input column names, most to blame for bad rows (bad on the goals) first."
  tbl   = ezr.Tbl(ezr.csv(train_csv))
  names = [tbl.names[c.at] for c in tbl.x]
  goals = _goals(tbl)
  if not goals or not tbl.x: return sorted(names)
  rows  = [r for r in tbl.rows if _complete(goals, r)]  # unrankable rows are skipped
  if not rows: return sorted(names)
  ys    = [_ydist(goals, r) for r in rows]
  cut   = sorted(ys)[min(len(ys) - 1, int(BAD * (len(ys) - 1)))]
  bad   = [y >= cut for y in ys]
  base  = sum(bad) / len(rows)
  total = sum(bad)
  scores: dict[str, float] = {}
  for col in tbl.x:
    name, best = tbl.names[col.at], 0.0
    for _, go in ezr.candidates(col, rows):
      yes   = [i for i, r in enumerate(rows) if go(r)]
      n_yes = len(yes)
      n_no  = len(rows) - n_yes
      if n_yes < ezr.the.Leaf or n_no < ezr.the.Leaf: continue
      bad_yes = sum(bad[i] for i in yes)
      ry = bad_yes / n_yes
      rn = (total - bad_yes) / n_no
      best = max(best, ry - base, rn - base)
    scores[name] = best
  return sorted(names, key=lambda n: (-scores[n], n))

