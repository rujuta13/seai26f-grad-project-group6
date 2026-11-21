# step9.py
#!/usr/bin/env python3 -B
"""
step9.py: spot.  Flag benchmark results that look like glitches, and flag
flag sets unlike anything seen before.  Each row gets a suspicion score;
higher means odder.  Three parts, summed:
  far     how far the inputs are from the nearest training flag set (ezr.xdist)
  unseen  inputs never seen in training: a symbol value absent from training,
          or a number outside the training range (in sd units)
  glitch  the worst goal, as a z-score against the training goals
Missing cells ('?') add nothing: they are blind, not odd.
Reuses ezr for distance and column stats, and step2 and step4 for the
cached table and row handling.
"""
from functools import lru_cache
import ezr
import step2, step4

@lru_cache(maxsize=None)
def _ranges(train_csv: str) -> dict[int, tuple[float, float]]:
  "Min and max of each numeric input column in the training rows."
  tbl = step2._tbl(train_csv)
  out = {}
  for c in tbl.x:
    if type(c) is ezr.Num:
      xs = [r[c.at] for r in tbl.rows if r[c.at] != "?"]
      if xs: out[c.at] = (min(xs), max(xs))
  return out

def _score(tbl, row: list, ranges: dict) -> float:
  "Suspicion of one ezr ROW: higher is odder."
  far    = min((ezr.xdist(tbl, row, t) for t in tbl.rows), default=0.0)
  unseen = 0.0
  for c in tbl.x:
    v = row[c.at]
    if v == "?": continue
    if type(c) is ezr.Sym:
      if v not in c.has: unseen += 1.0
    elif c.at in ranges:
      lo, hi = ranges[c.at]
      unseen += max(0.0, lo - v, v - hi) / (c.sd or 1)
  glitch = 0.0
  for c in tbl.y:
    v = row[c.at]
    if v == "?": continue
    glitch = max(glitch, abs(v - c.mu) / (c.sd or 1))
  return float(far + unseen + glitch)

def odd(train_csv: str, rows) -> list[float]:
  "One suspicion score per row in ROWS, goals included; higher = odder."
  tbl    = step2._tbl(train_csv)
  ranges = _ranges(train_csv)
  return [_score(tbl, step4._row(tbl, r), ranges) for r in rows]

