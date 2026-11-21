# step9.py
from math import log1p
import ezr
import step2

def odd(train_csv: str, rows) -> list[float]:
  "One suspicion score per row in ROWS (goals included): higher means more likely a glitch, a typo, or unlike anything in train_csv."
  tbl = step2._table(train_csv)
  if not tbl.rows: raise ValueError(f"no rows in {train_csv}")
  out = []
  for row in rows:
    full = step2._row(tbl, row)
    # inputs: distance to the nearest train row (impossible or unseen combinations)
    s = min(ezr.xdist(tbl, full, t) for t in tbl.rows)
    for c in tbl.x:
      v = full[c.at]
      if v == "?": continue
      if type(c) is ezr.Sym:
        if v not in c.has: s += 1.0            # a symbol never seen in train: likely a typo
      elif not isinstance(v, (int, float)):
        s += 1.0                               # text in a numeric column: a glitch
      else:                                    # numbers far outside train's spread
        excess = abs(v - c.mu) / (c.sd + 1e-32) - 3
        s += log1p(max(0.0, excess))
    # goals: a missing measurement, or one far outside train's spread
    for c in tbl.y:
      v = full[c.at]
      if v == "?":
        s += 1.0                               # no measurement recorded
      elif not isinstance(v, (int, float)):
        s += 1.0                               # text where a number belongs
      else:
        excess = abs(v - c.mu) / (c.sd + 1e-32) - 3
        s += log1p(max(0.0, excess))
    out.append(float(s))
  return out

