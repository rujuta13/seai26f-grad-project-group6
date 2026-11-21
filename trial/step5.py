# step5.py
"""Blame: which input columns explain the bad rows (bad on the goals).

Bad rows are the worst quarter by goal distance (step4's ydist over
goals only). For each input (x) column, blame is how much more often
one of its values shows up among bad rows than among all rows: the
largest gain P(value | bad) - P(value | all). Numeric columns are split
at their median first, so each has two values. Columns come back most
to blame first; ties keep header order.
"""
import ezr
import step2
import step4

BAD = 0.75   # rows at or above this ydist percentile count as bad


def _values(col, rows, med) -> list:
  "Each row's value for COL; numbers become above/below MED, '?' stays."
  out = []
  for r in rows:
    v = r[col.at]
    if v == "?":
      out.append("?")
    elif type(col) is ezr.Sym:
      out.append(v)
    else:
      out.append(v <= med)
  return out


def blame(train_csv: str) -> list[str]:
  "Input column names, most to blame for bad rows (bad on the goals) first."
  tbl  = step2._load(train_csv)
  rows = tbl.rows
  ys   = [step4._ydist(tbl, r) for r in rows]
  cut  = step2._pct(ys, BAD)
  bad  = [r for r, y in zip(rows, ys) if y >= cut]

  def lift(col) -> float:
    "Largest gain in a value's share among bad rows over its share overall."
    med = 0
    if type(col) is ezr.Num:
      known = [r[col.at] for r in rows if step2._known(r[col.at])]
      med = step2._pct(known, .5) if known else 0
    vals_all = [v for v in _values(col, rows, med) if v != "?"]
    vals_bad = [v for v in _values(col, bad, med) if v != "?"]
    if not vals_all or not vals_bad: return 0.0
    best = 0.0
    for v in set(vals_all):
      p_all = vals_all.count(v) / len(vals_all)
      p_bad = vals_bad.count(v) / len(vals_bad)
      best  = max(best, p_bad - p_all)
    return best

  ranked = sorted(tbl.x, key=lambda c: -lift(c))   # stable: ties keep order
  return [c.txt for c in ranked]

