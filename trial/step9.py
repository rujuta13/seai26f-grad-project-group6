#!/usr/bin/env python3 -B
"step9.py: spot. Which benchmark results look like glitches, and which flag sets behave unlike anything seen?"
from math import isfinite
from ezr import Sym, xdist
from step2 import _quantile
from step4 import _base, _row

Z_OK         = 3.0    # a goal this many sds from train's mean is still normal
W_MISS_GOAL  = 1.0    # a missing measurement is a suspicious result
W_MISS_INPUT = 0.5    # a missing flag is less alarming
W_UNSEEN     = 1.0    # an input value train never shows
W_FAR        = 1.0    # distance beyond train's own typical spacing
Q_FAR        = 0.95   # train's own spread: this quantile of nearest-neighbour distances
MIN_SCALE    = 0.05   # floor on that spread, so tiny spreads do not explode

def odd(train_csv, rows) -> list[float]:
  """One suspicion score per row in ROWS (goals included): higher means more
  likely a glitch, a typo, or unlike anything in TRAIN_CSV.  Each row scores
  missing cells, input values train never shows, goals far outside train's
  spread, and inputs far from every train row compared with train's own spacing."""
  base, tbl = _base(str(train_csv))
  q95  = max(_quantile(base, Q_FAR), MIN_SCALE) if base else MIN_SCALE
  goal_ids = {id(c) for c in tbl.y}
  cols = list(tbl.cols.values())
  out  = []
  for row in rows:
    full = _row(tbl, row)
    s    = 0.0
    for c in cols:
      v = full[c.at]
      if v == "?":                                 # a missing cell
        s += W_MISS_GOAL if id(c) in goal_ids else W_MISS_INPUT
        continue
      if type(c) is Sym:
        if v not in c.has: s += W_UNSEEN           # a value train never shows
      else:                                        # a Num column
        if not isinstance(v, (int, float)) or not isfinite(v):
          s += W_UNSEEN
        elif c.sd == 0:                            # constant in train
          s += 1.0 if v != c.mu else 0.0
        else:                                      # how many sds beyond normal
          s += max(0.0, abs(v - c.mu) / c.sd - Z_OK)
    # inputs: how far is this row from its nearest train row?
    nn  = min((xdist(tbl, full, q) for q in tbl.rows), default=0)
    s  += W_FAR * max(0.0, nn - q95) / q95
    out.append(float(s))
  return out

