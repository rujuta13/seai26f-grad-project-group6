# step5.py
"Step 5: blame. Which input columns are most to blame for bad rows?"
import ezr

MIN  = 2     # smallest group (rows) a split may isolate
CUTS = 4     # thresholds tried per numeric column

def _ydist(tbl, row):
  "Distance of ROW's numeric goals from their best; None if no goal is known."
  sq, n = 0.0, 0
  for c in tbl.y:
    if type(c) is not ezr.Num: continue
    v = row[c.at]
    if v == "?": continue
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
  tbl  = ezr.Tbl(ezr.csv(train_csv))
  seen = []
  for r in tbl.rows:
    d = _ydist(tbl, r)
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

