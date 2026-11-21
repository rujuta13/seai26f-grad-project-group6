#!/usr/bin/env python3 -B
"step5.py: blame. Which input columns best separate the bad rows from the good ones?"
from ezr import Tbl, csv, ydist, candidates
from step1 import best

BAD = 0.25   # worst fraction of rows, by ydist, counted as bad

def blame(train_csv) -> list[str]:
  "Input column names, most to blame for bad rows (bad on the goals) first."
  tbl  = Tbl(csv(str(train_csv)))
  rows = tbl.rows
  n    = len(rows)
  if n == 0: return []
  good_ix = set(best(str(train_csv)))
  ranked  = sorted(range(n), key=lambda i: ydist(tbl, rows[i]), reverse=True)
  bad_ix  = ranked[:max(1, int(BAD * n))]
  good    = [rows[i] for i in good_ix]
  bad     = [rows[i] for i in bad_ix if i not in good_ix]
  if not good or not bad: return [tbl.names[c.at] for c in tbl.x]

  def score(col) -> float:
    "The biggest gap, over this column's splits, between bad and good rows that fall on the split's yes side."
    gap = 0.0
    for _, go in candidates(col, rows):
      pb = sum(1 for r in bad  if go(r)) / len(bad)
      pg = sum(1 for r in good if go(r)) / len(good)
      gap = max(gap, pb - pg)
    return gap

  scored = [(score(c), tbl.names[c.at]) for c in tbl.x]
  scored.sort(key=lambda t: -t[0])   # stable: ties keep header order
  return [name for _, name in scored]

