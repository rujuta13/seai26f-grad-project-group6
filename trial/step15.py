# step15.py
"Step 15: blame. Which input columns are most to blame for bad rows? Judged by all numeric goals, or by the goals named in FOCUS (e.g. FOCUS = ("Penalties",) for weak-from-the-spot, FOCUS = ("Strength",) for physical strength)."
import math
import ezr

MIN   = 2      # smallest group (rows) a split may isolate
CUTS  = 4      # thresholds tried per numeric column
FOCUS = ()     # goal names to judge by; () means every numeric goal

def _isnum(v) -> bool:
  "True for a usable finite number."
  return (isinstance(v, (int, float)) and not isinstance(v, bool)
          and math.isfinite(v))

def _goals(tbl) -> list:
  "The numeric goals to judge by: those named in FOCUS, or all of them if FOCUS is empty."
  nums = [c for c in tbl.y if type(c) is ezr.Num]
  if not FOCUS: return nums
  want   = {f.lower() for f in FOCUS}
  picked = [c for c in nums
            if c.txt.lower() in want or c.txt[:-1].lower() in want]
  if not picked:
    raise ValueError(f"blame: no numeric goal named in {FOCUS!r}")
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

def blame(train_csv) -> list[str]:
  "Input column names, most to blame for bad rows (bad on the goals) first."
  tbl   = ezr.Tbl(ezr.csv(train_csv))
  goals = _goals(tbl)
  seen  = []
  for r in tbl.rows:
    d = _ydist(goals, r)
    if d is not None: seen.append((r, d))
  score = {}
  for col in tbl.x:
    vals = [(r[col.at], d) for r, d in seen if r[col.at] != "?"]
    best = 0.0
    if vals:
      for _, test in _splits(col, [x for x, _ in vals]):
        yes = [d for x, d in vals if test(x)]
        no  = [d for x, d in vals if not test(x)]
        if len(yes) < MIN or len(no) < MIN: continue
        gap = sum(yes) / len(yes) - sum(no) / len(no)
        best = max(best, gap * len(yes) ** .5)
    score[col.txt] = best
  return sorted(score, key=lambda k: -score[k])

