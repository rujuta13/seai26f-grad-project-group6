# step25.py
"Step 25: blame. Which input columns are most to blame for customers who never respond to campaigns (RESPONSE) or who complain (COMPLAIN)?"
import math
import ezr

MIN   = 2                          # smallest group (rows) a split may isolate
CUTS  = 4                          # thresholds tried per numeric column
GOALS = ("RESPONSE", "COMPLAIN")   # the goals the client asked about

def _name(s: str) -> str:
  "A column name without its +, - or ! suffix."
  return s[:-1] if s[-1:] in ("+", "-", "!") else s

def _isnum(v) -> bool:
  "True for a usable finite number."
  return (isinstance(v, (int, float)) and not isinstance(v, bool)
          and math.isfinite(v))

def _usable(col, v) -> bool:
  "True if cell V of column COL can take part in a split."
  if type(col) is ezr.Sym: return v != "?"
  return _isnum(v)

def _goals(tbl) -> list:
  "The numeric goals named in GOALS, each judged on its own."
  picked = [c for c in tbl.y
            if type(c) is ezr.Num and _name(c.txt).upper() in GOALS]
  if not picked:
    raise ValueError(f"blame: no numeric goal named in {GOALS!r}")
  return picked

def _ydist(goals, row) -> float | None:
  "Distance of ROW's numeric GOALS from their best; None if no goal is known."
  sq, n = 0.0, 0
  for c in goals:
    v = row[c.at]
    if not _isnum(v): continue
    sq += (ezr.norm(c, v) - c.goal) ** 2
    n  += 1
  return (sq / n) ** .5 if n else None

def _splits(col, xs: list):
  "Yield (label, test) pairs: one per symbol value, or CUTS thresholds per number."
  if type(col) is ezr.Sym:
    for v in sorted(set(xs), key=str):
      yield v, (lambda x, v=v: x == v)
  else:
    ys = sorted(xs)
    for i in range(1, CUTS + 1):
      t = ys[(i * (len(ys) - 1)) // (CUTS + 1)]
      yield t, (lambda x, t=t: x <= t)
      yield t, (lambda x, t=t: x > t)

def _score(col, seen: list) -> float:
  "How much the worst split of COL's values isolates worse goal distances (0 if none)."
  vals = [(r[col.at], d) for r, d in seen if _usable(col, r[col.at])]
  best = 0.0
  if not vals: return best
  for _, test in _splits(col, [x for x, _ in vals]):
    yes = [d for x, d in vals if test(x)]
    no  = [d for x, d in vals if not test(x)]
    if len(yes) < MIN or len(no) < MIN: continue
    gap = sum(yes) / len(yes) - sum(no) / len(no)
    best = max(best, gap * len(yes) ** .5)
  return best

def blame(train_csv) -> list[str]:
  "Input column names, most to blame for bad rows (bad on the goals) first."
  tbl   = ezr.Tbl(ezr.csv(train_csv))
  goals = _goals(tbl)
  score = {col.txt: 0.0 for col in tbl.x}
  for g in goals:
    seen = []
    for r in tbl.rows:
      d = _ydist([g], r)
      if d is not None: seen.append((r, d))
    for col in tbl.x:
      score[col.txt] = max(score[col.txt], _score(col, seen))
  return sorted(score, key=lambda k: (-score[k], k))

