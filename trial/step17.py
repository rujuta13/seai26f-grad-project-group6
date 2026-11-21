# step17.py: choose. Find the best row while labelling at most BUDGET rows.
"step17.py: choose. Find the best row while labelling at most BUDGET rows."
import math
import random
import ezr

START = 4      # initial random labels before the search starts

def _pick_goals(tbl, goal):
  "The goal columns to optimise: all of them, or only the one named GOAL."
  if not tbl.y: raise ValueError("no goal columns (names ending in + or -)")
  if goal is None: return list(tbl.y)
  want = str(goal).lower()
  for c in tbl.y:
    if want in (c.txt.lower(), c.txt[:-1].lower()): return [c]
  raise ValueError(f"no goal named {goal!r}; goals are {[c.txt for c in tbl.y]}")

def _value(d, c):
  "The label's value for goal column C, by full name or without its suffix; '?' if absent."
  for key in (c.txt, c.txt[:-1]):
    if key in d:
      v = d[key]
      return ezr.atom(v) if isinstance(v, str) else v
  return "?"

def _score(lab, row, ats):
  "How far ROW's goals (columns ATS) are from ideal, 0 is best; stats come from LAB."
  cols = [lab.cols[at] for at in ats]
  return ezr.dist((abs(ezr.norm(c, row[c.at]) - c.goal) for c in cols),
                  len(cols))

def choose(blind_csv, label, budget=30, goal=None) -> int:
  """Index of the best row found, labelling at most BUDGET rows.
  BLIND_CSV has every goal cell set to '?'. LABEL(i) gives row i's goals as a dict.
  GOAL, if given, names the one goal to optimise; otherwise all goals are optimised."""
  tbl  = ezr.Tbl(ezr.csv(blind_csv))
  rows = tbl.rows
  n    = len(rows)
  if n == 0: raise ValueError("no rows to choose from")
  if budget < 1: raise ValueError("budget must be at least 1")
  goals = _pick_goals(tbl, goal)
  ats   = [c.at for c in goals]
  rng   = random.Random(ezr.the.Seed)
  used  = 0

  def ask(i):
    "Label row I, counting against the budget."
    nonlocal used
    if used >= budget: raise RuntimeError(f"budget of {budget} labels exceeded")
    used += 1
    d   = label(i)
    row = list(rows[i])
    for c in tbl.y: row[c.at] = _value(d, c)
    for c in goals:
      if row[c.at] == "?": raise ValueError(f"label({i}) gave no value for {c.txt!r}")
    return row

  order = rng.sample(range(n), n)
  start = min(START, budget, n)
  ids   = order[:start]          # rows labelled so far, in order
  todo  = order[start:]          # rows not yet labelled
  done  = [ask(i) for i in ids]  # their full rows, goals filled in

  while todo and used < budget:
    lab    = ezr.Tbl([tbl.names] + done)
    ranked = sorted(range(len(done)), key=lambda k: _score(lab, done[k], ats))
    cut    = max(1, int(math.sqrt(len(done))))     # the best stay few
    best   = [done[k] for k in ranked[:cut]]
    rest   = [done[k] for k in ranked[cut:]]
    cb     = ezr.mids(ezr.clone(tbl, best))
    if rest:
      cr = ezr.mids(ezr.clone(tbl, rest))
      # near the best centroid, far from the rest centroid
      todo.sort(key=lambda i: ezr.xdist(tbl, rows[i], cr)
                              - ezr.xdist(tbl, rows[i], cb))
    else:
      todo.sort(key=lambda i: -ezr.xdist(tbl, rows[i], cb))
    i = todo.pop()
    ids.append(i)
    done.append(ask(i))

  lab    = ezr.Tbl([tbl.names] + done)
  scores = [_score(lab, r, ats) for r in done]
  return ids[min(range(len(done)), key=lambda k: scores[k])]

