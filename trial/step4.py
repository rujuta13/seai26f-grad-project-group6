# step4.py
#!/usr/bin/env python3 -B
"""
step4.py: flow.  Benchmark rows arrive one at a time.  stream() keeps a row
if its goals are within the best 5% of the training rows (step1.best).
drift() warns at the first row where the last WINDOW goal values stop
looking like the training goal values, by both Cliff's delta and KS.
Reuses step1 (best rows), step2 (cached table, probe), and ezr (distance,
tests).  Rows are never added to the training table, so only train_csv and
earlier rows ever inform a decision.
"""
from functools import lru_cache
from collections.abc import Iterable
import ezr
import step1, step2

WINDOW = 30   # drift: how many recent rows must look different

def _row(tbl, r) -> list:
  "An arriving row as an ezr row in table order.  A dict is matched by header name."
  if isinstance(r, dict): return step2._probe(tbl, r)
  return list(r)

@lru_cache(maxsize=None)
def _cut(train_csv: str) -> float:
  "ydist of the worst of the best 5% training rows: the keep threshold."
  tbl = step2._tbl(train_csv)
  idx = step1.best(train_csv)
  return max((ezr.ydist(tbl, tbl.rows[i]) for i in idx), default=0.0)

@lru_cache(maxsize=None)
def _goals(train_csv: str) -> dict[str, list]:
  "Sorted training values of each goal column, keyed by goal name."
  tbl = step2._tbl(train_csv)
  return {c.txt: sorted(r[c.at] for r in tbl.rows if r[c.at] != "?")
          for c in tbl.y}

def stream(train_csv: str, rows: Iterable) -> list[bool]:
  "Keep (True) each arriving row whose goals are within the training best 5%."
  tbl = step2._tbl(train_csv)
  cut = _cut(train_csv)
  out = []
  for r in rows:
    row = _row(tbl, r)
    ok  = (all(row[c.at] != "?" for c in tbl.y)
           and ezr.ydist(tbl, row) <= cut)
    out.append(bool(ok))
  return out

def drift(train_csv: str, rows: Iterable) -> int | None:
  "Index of the first row whose last WINDOW goal values differ from training."
  tbl  = step2._tbl(train_csv)
  ref  = _goals(train_csv)
  seen = {c.txt: [] for c in tbl.y}
  for i, r in enumerate(rows):
    row = _row(tbl, r)
    for c in tbl.y: seen[c.txt].append(row[c.at])
    if i + 1 < WINDOW: continue
    for c in tbl.y:
      win = sorted(v for v in seen[c.txt][-WINDOW:] if v != "?")
      if not win: continue
      # different = both tests reject "same", which avoids warning on noise
      if not ezr.cliffs(win, ref[c.txt]) and not ezr.ks(win, ref[c.txt]):
        return i
  return None

