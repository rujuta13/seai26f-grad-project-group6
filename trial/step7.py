#!/usr/bin/env python3 -B
"step7.py: choose. Find the best flag set while benchmarking at most BUDGET builds."
from ezr import Tbl, acquire, clone, csv, the, ydist

def choose(blind_csv, label, budget=30, goal=None) -> int:
  """Index of the best row, labelling at most BUDGET rows.  BLIND_CSV has
  every goal '?'; LABEL(i) gives row i's goals as a dict.  GOAL, if given,
  optimises that one goal only."""
  if budget < 1: raise ValueError("budget must be at least 1")
  rows  = list(csv(str(blind_csv)))
  if len(rows) < 2: raise ValueError(f"no rows in {blind_csv}")
  names = rows[0]
  goals = [n for n in names if n[-1] in "+-"]
  keep  = None
  if goal is not None:
    want = goal.rstrip("+-").lower()
    hits = [n for n in goals if n[:-1].lower() == want]
    if not hits: raise ValueError(f"no goal named {goal!r}; goals are {goals}")
    keep  = hits[0]
    # ezr skips any column whose name ends in X, so the other goals vanish
    names = [n[:-1] + "X" if n in goals and n != keep else n for n in names]
  tbl    = Tbl([names] + rows[1:])
  pool   = tbl.rows
  where  = {id(r): i for i, r in enumerate(pool)}  # row object -> file index
  budget = min(budget, len(pool))
  got, calls = {}, [0]

  def labeller(row):
    "Ask LABEL for this row's goals, once per row, and never past BUDGET."
    if calls[0] >= budget: raise RuntimeError(f"more than {budget} labels asked")
    calls[0] += 1
    i    = where[id(row)]
    val  = label(i)
    full = list(row)
    for c in tbl.y:
      key = c.txt if c.txt in val else c.txt[:-1]
      full[c.at] = val[key]
    got[i] = full
    return full

  saved = the.Start, the.Few
  the.Start, the.Few = min(the.Start, budget), len(pool)  # sample the whole pool
  try:
    acquire(tbl, budget, labeller)
  finally:
    the.Start, the.Few = saved
  lab = clone(tbl, list(got.values()))
  return min(got, key=lambda i: ydist(lab, got[i]))

