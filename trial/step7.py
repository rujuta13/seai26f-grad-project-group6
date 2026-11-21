# step7.py
#!/usr/bin/env python3 -B
"Choose the best flag set while benchmarking at most BUDGET builds, via ezr."
import random
import ezr

def choose(blind_csv, label, budget=30, goal=None) -> int:
  """Index of the best build found, labelling at most BUDGET rows.
  BLIND_CSV has every goal cell '?'. LABEL(i) returns the goals of row i as
  a {goal name: value} dict. If GOAL names one goal, only that goal counts;
  it may omit the +/- suffix and ignores case."""
  tbl  = ezr.Tbl(ezr.csv(blind_csv))
  rows = tbl.rows
  if not rows: raise ValueError("no rows to choose from")
  all_goals = [c for c in tbl.y if c.txt[-1] in "+-"]
  if not all_goals: raise ValueError("the table has no goal columns")
  if goal is None:
    goals = all_goals
  else:
    hits = [c for c in all_goals if c.txt == goal]
    if not hits:
      stem = goal.rstrip("+-").lower()
      hits = [c for c in all_goals if c.txt[:-1].lower() == stem]
    if len(hits) != 1:
      names = ", ".join(c.txt for c in all_goals)
      raise ValueError(f"goal {goal!r} must name exactly one of: {names}")
    goals = hits
  if budget < 1: raise ValueError("budget must be at least 1")
  budget = min(int(budget), len(rows))

  def value(v):
    "One goal cell as a number, or '?' if unknown."
    if isinstance(v, str): v = ezr.atom(v.strip() or "?")
    if v is None or v == "?": return "?"
    if not isinstance(v, (int, float)):
      raise ValueError(f"goal value {v!r} is not a number")
    return v

  known = {}                              # row index -> {column at: goal cell}

  def ask(i):
    "Label row I. Never more than BUDGET labels are taken."
    if len(known) >= budget: raise RuntimeError("label budget exhausted")
    got = label(i)
    cells = {}
    for c in goals:
      if c.txt not in got:
        raise KeyError(f"label({i}) gave no value for {c.txt!r}")
      cells[c.at] = value(got[c.txt])
    known[i] = cells

  def scores() -> dict:
    "Each labelled row's goal distance from the ideal; inf if a goal is unknown."
    cols = {c.at: ezr.Num(c.txt, c.at) for c in goals}   # stats of labels only
    for cells in known.values():
      for c in goals: ezr.add(cols[c.at], cells[c.at])
    out = {}
    for i, cells in known.items():
      if any(cells[c.at] == "?" for c in goals):
        out[i] = float("inf")
      else:
        out[i] = ezr.dist((abs(ezr.norm(cols[c.at], cells[c.at])
                               - cols[c.at].goal) for c in goals),
                          len(goals))
    return out

  rng   = random.Random(ezr.the.Seed)
  order = rng.sample(range(len(rows)), len(rows))
  start = min(budget, max(1, int(ezr.the.Start)))
  for i in order[:start]: ask(i)
  todo = order[start:]

  while len(known) < budget and todo:
    sc     = scores()
    ranked = sorted(known, key=lambda i: (sc[i], i))
    k      = max(1, int(len(ranked) ** 0.5))     # the best, about sqrt(n)
    cb     = ezr.mids(ezr.clone(tbl, [rows[i] for i in ranked[:k]]))
    rest   = [rows[i] for i in ranked[k:]]
    if not tbl.x:
      pick = todo[0]
    elif rest:
      cr   = ezr.mids(ezr.clone(tbl, rest))
      # near the best labelled rows, far from the rest (as in acquire)
      pick = max(todo, key=lambda i: ezr.xdist(tbl, rows[i], cr)
                                     - ezr.xdist(tbl, rows[i], cb))
    else:
      pick = min(todo, key=lambda i: ezr.xdist(tbl, rows[i], cb))
    todo.remove(pick)
    ask(pick)

  sc = scores()
  return min(known, key=lambda i: (sc[i], i))

