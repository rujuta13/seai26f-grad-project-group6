# step7.py
"""
step7: choose.  Find the best flag set while labelling at most BUDGET
builds.  The pool arrives blind (every goal cell '?'); label(i) reveals
the goals of row i.  Reuses ezr's table, clone, acquire, and ydist.
With GOAL named, only that goal counts: the other goals are renamed to
end in X, which makes ezr skip them.
"""
import ezr


def choose(blind_csv, label, budget=30, goal=None) -> int:
  "Index of the best row, labelling at most BUDGET rows. GOAL, if given, is the only goal."
  if budget < 1:
    raise ValueError("choose: budget must be at least 1")
  raw   = list(ezr.csv(blind_csv))
  full  = [str(s) for s in raw[0]]            # header names, as written
  data  = raw[1:]
  if not data:
    raise ValueError("choose: no rows to choose from")
  goals = [at for at, s in enumerate(full) if s.endswith(("+", "-", "!"))]

  pick = None
  if goal is not None:
    hits = [at for at in goals if goal in (full[at], full[at][:-1])]
    if not hits:
      raise ValueError(f"choose: no goal named {goal!r}")
    pick = hits[0]
  # Other goals end in X, so ezr skips them: only PICK drives ydist.
  names = ([s if (at == pick or at not in goals) else s[:-1] + "X"
            for at, s in enumerate(full)]
           if pick is not None else full)

  tbl   = ezr.Tbl(iter([names] + data))
  where = {id(r): i for i, r in enumerate(tbl.rows)}   # row object -> index

  def cell(v):
    "Strings become ATOMs; numbers pass through."
    return ezr.atom(v) if isinstance(v, str) else v

  def read(got, name):
    "The value of one goal in GOT, by full name or without its sign."
    for key in (name, name[:-1]):
      if key in got: return got[key]
    raise KeyError(f"choose: label gave no value for {name!r}")

  calls = 0
  made  = {}                                   # id(labelled row) -> (i, row)

  def oracle(row):
    "Label ROW's source index, and return ROW with its goal cells filled."
    nonlocal calls
    if calls >= budget:
      raise RuntimeError(f"choose: label budget of {budget} exceeded")
    calls += 1
    i   = where[id(row)]
    got = label(i)
    out = list(row)
    for at in goals:
      out[at] = cell(read(got, full[at]))
    made[id(out)] = (i, out)
    return out

  old_start, old_few = ezr.the.Start, ezr.the.Few
  ezr.the.Start = max(1, min(ezr.the.Start, budget))  # never over budget
  ezr.the.Few   = len(data)                           # every row is a candidate
  try:
    got = ezr.acquire(tbl, cap=budget, label=oracle)
  finally:
    ezr.the.Start, ezr.the.Few = old_start, old_few  # always put `the` back

  lab  = ezr.clone(tbl, got)                   # stats from labelled rows only
  best = min(got, key=lambda r: ezr.ydist(lab, r))
  return made[id(best)][0]

