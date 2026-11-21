# step7.py
#!/usr/bin/env python3 -B
"""
step7.py: choose.  Find the best flag set while benchmarking at most BUDGET
builds.  The blind table has every goal cell set to '?'; label(i) reveals
the goals of row i.  choose() spends its budget with ezr.acquire (label
builds near the best so far, far from the rest), then returns the best
labelled row.  With GOAL set, only that goal counts: the other goal columns
are renamed to 'X' (skipped) in a private copy of the table, so ezr sees
one goal and every distance and cut uses only it.
"""
import ezr

def _goal_at(names: list, key: str) -> int | None:
  "Column index of the goal named KEY, with or without its +, -, ! suffix."
  for at, s in enumerate(names):
    if s[-1] in "+-!" and key in (s, s[:-1]): return at
  return None

def choose(blind_csv: str, label, budget: int = 30,
           goal: str | None = None) -> int:
  "Index of the best row found, labelling at most BUDGET rows via LABEL."
  if budget < 1: raise ValueError("budget must be at least 1")
  src   = list(ezr.csv(blind_csv))
  names = src[0]
  goals = [at for at, s in enumerate(names) if s[-1] in "+-!"]
  if not goals: raise ValueError(f"no goal columns in {names}")
  if goal is None:
    want, keep = goals, names
  else:
    target = _goal_at(names, goal)
    if target is None: raise ValueError(f"no goal named {goal!r} in {names}")
    want = [target]
    keep = [s[:-1] + "X" if at in goals and at != target else s
            for at, s in enumerate(names)]
  tbl = ezr.Tbl([keep] + src[1:])
  if not tbl.rows: raise ValueError("no rows to choose from")

  index = {id(r): i for i, r in enumerate(tbl.rows)}  # row object -> index
  back  = {}                                          # labelled row -> index
  spent = [0]

  def oracle(row):
    "Reveal the goals of ROW by calling the client's LABEL, once per row."
    if spent[0] >= budget: raise RuntimeError(f"budget of {budget} labels spent")
    spent[0] += 1
    i      = index[id(row)]
    got    = label(i)
    filled = list(row)
    for key, v in got.items():
      at = _goal_at(names, key)
      if at is None: raise KeyError(f"label({i}) gave unknown goal {key!r}")
      filled[at] = ezr.atom(v) if isinstance(v, str) else v
    for at in want:
      if filled[at] == "?": raise KeyError(f"label({i}) gave no value for {names[at]}")
    back[id(filled)] = i
    return filled

  old_start, old_few = ezr.the.Start, ezr.the.Few
  ezr.the.Start = min(old_start, budget, len(tbl.rows))  # never exceed budget
  ezr.the.Few   = len(tbl.rows)                          # consider every row
  try:
    labelled = ezr.acquire(tbl, budget, oracle)
  finally:
    ezr.the.Start, ezr.the.Few = old_start, old_few

  scorer = ezr.Tbl([keep] + labelled)   # normalises over what we labelled
  best   = min(labelled, key=lambda r: ezr.ydist(scorer, r))
  return back[id(best)]

