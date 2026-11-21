#!/usr/bin/env python3 -B
"step4.py: flow. Keep the good benchmark rows as they arrive; warn when they stop looking like train."
from collections import deque
from functools import lru_cache
from ezr import Tbl, csv, xdist, ydist, ks
from step1 import best

W      = 40     # recent new rows compared, as a window, against train
A      = 1.95   # ks threshold: about a 0.1% chance of a false alarm per test
SAMPLE = 200    # train rows used as the baseline for drift

def _row(tbl, row) -> list:
  """ROW as a full table row, in header order.  ROW is a dict, a full
  list, or a list in header order minus goals (missing cells are '?')."""
  names = tbl.names
  if isinstance(row, dict):
    return [row.get(s, "?") for s in names]
  row = list(row)
  if len(row) == len(names): return row
  free = [s for s in names if s[-1] not in "+-"]
  got  = dict(zip(free, row))
  return [got.get(s, "?") for s in names]

@lru_cache(maxsize=4)
def _ref(train_csv: str):
  "The training table, and the ydist cutoff: the worst ydist among the best rows."
  tbl = Tbl(csv(train_csv))
  if not tbl.rows: raise ValueError(f"no rows in {train_csv}")
  idx    = best(train_csv)
  ys     = [ydist(tbl, r) for r in tbl.rows]
  cutoff = max(ys[i] for i in idx) if idx else min(ys)
  return tbl, cutoff

@lru_cache(maxsize=4)
def _base(train_csv: str):
  """Train's own spread: each sampled train row's distance to its
  nearest other train row, sorted.  Returns (base, tbl)."""
  tbl, _ = _ref(train_csv)
  rows   = tbl.rows
  step   = max(1, len(rows) // SAMPLE)
  base   = [min((xdist(tbl, rows[i], q)
                 for j, q in enumerate(rows) if j != i), default=0)
            for i in range(0, len(rows), step)]
  return sorted(base), tbl

def stream(train_csv, rows) -> list[bool]:
  """Keep a row if its goals are at least as good as the best rows of
  train.  Uses only train_csv and the row in hand, so it is causal."""
  tbl, cutoff = _ref(str(train_csv))
  out = []
  for row in rows:
    full = _row(tbl, row)
    if any(full[c.at] == "?" for c in tbl.y):
      out.append(False)              # goals unknown: cannot judge it good
      continue
    out.append(ydist(tbl, full) <= cutoff)
  return out

def drift(train_csv, rows) -> int | None:
  """Index of the first row at which the last W rows' distances to train
  stop looking like train's own distances.  None if they never do."""
  base, tbl = _base(str(train_csv))
  recent = deque(maxlen=W)
  for i, row in enumerate(rows):
    me = _row(tbl, row)
    recent.append(min((xdist(tbl, me, q) for q in tbl.rows), default=0))
    if len(recent) == W and not ks(base, sorted(recent), A):
      return i
  return None

