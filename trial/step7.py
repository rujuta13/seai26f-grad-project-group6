# step7.py
import random
import ezr

def choose(blind_csv, label, budget=30, goal=None) -> int:
  "Index of the best row, labelling at most BUDGET rows with LABEL. GOAL names one goal to optimise."
  if budget < 1: raise ValueError("choose: budget must be at least 1")
  src     = list(ezr.csv(blind_csv))
  headers = list(src[0])
  names   = list(headers)
  rows    = [list(r) for r in src[1:]]
  if not rows: raise ValueError(f"no rows in {blind_csv}")

  goals = [at for at, s in enumerate(headers) if str(s)[-1] in "+-"]
  if goal is not None:                       # one goal: the others become ignored
    hit = [at for at in goals
           if headers[at] == goal or headers[at].rstrip("+-") == goal]
    if len(hit) != 1: raise ValueError(f"choose: no single goal named {goal!r}")
    for at in goals:
      if at != hit[0]: names[at] = headers[at][:-1] + "X"

  tbl   = ezr.Tbl([names] + rows)
  at_of = {id(r): i for i, r in enumerate(rows)}   # row object -> row index
  spent = [0]

  def oracle(row):                           # fills the goal cells of ROW, in place
    if spent[0] >= budget:
      raise RuntimeError(f"choose: label budget of {budget} exceeded")
    spent[0] += 1
    got = label(at_of[id(row)])
    for c in tbl.y:
      key = headers[c.at]
      if key not in got: raise ValueError(f"label gave no value for {key!r}")
      v = got[key]
      row[c.at] = ezr.atom(v) if isinstance(v, str) else v
    return row

  random.seed(ezr.the.Seed)                  # reproducible picks
  keep = ezr.the.Start, ezr.the.Few
  ezr.the.Start = min(ezr.the.Start, budget, len(rows))
  ezr.the.Few   = len(rows)                  # every row is a candidate
  try:
    labelled = ezr.acquire(tbl, budget, oracle)
  finally:
    ezr.the.Start, ezr.the.Few = keep

  done = ezr.clone(tbl, labelled)
  best = min(labelled, key=lambda r: ezr.ydist(done, r))
  return at_of[id(best)]

