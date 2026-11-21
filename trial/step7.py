# step7.py
"""Choose the best flag set while benchmarking at most BUDGET builds.

choose() sees only the flag sets (blind_csv has every goal cell '?').
It labels a few random builds, then keeps labelling the builds that
look most like the best-so-far and least like the rest. It is the
same loop as ezr.acquire, but the oracle is label(i), it stops at
BUDGET, and it returns a row index. With goal=..., only that one goal
is optimised.
"""
import random
from math import sqrt

import ezr
import step2
import step4

START = 4                                 # random labels before steering
_ALIAS = {"speed": "Performance", "cpu": "Cpu"}


def _goals_for(lab, goal):
  "All goal columns of LAB, or just the one named GOAL."
  gs = step4._goals(lab)
  if not gs: raise ValueError("choose: no goal columns (+ or -) in data")
  if goal is None: return gs
  name = _ALIAS.get(goal.lower(), goal).lower()
  hits = [g for g in gs
          if name in (g.txt.lower(), g.txt.rstrip("+-").lower())]
  if not hits:
    raise ValueError(f"no goal named {goal!r}; have {[g.txt for g in gs]}")
  return hits[:1]


def _value(d: dict, name: str):
  "The label dict's value for goal NAME (with or without its +/- suffix)."
  if name in d: return d[name]
  return d.get(name.rstrip("+-"), "?")


def _fill(tbl, row, d: dict) -> list:
  "A copy of blind ROW with its goal cells taken from label dict D."
  full = list(row)
  for g in step4._goals(tbl):
    v = _value(d, g.txt)
    full[g.at] = "?" if v is None else step2._cell(v)
  return full


def _cost(lab, row, goals) -> float:
  "How far ROW's chosen goals are from ideal; an unknown goal is far."
  return ezr.dist((abs(ezr.norm(g, row[g.at]) - g.goal)
                   if step2._known(row[g.at]) else 1
                   for g in goals), len(goals))


def choose(blind_csv, label, budget=30, goal=None) -> int:
  "Index of the best row found, labelling at most BUDGET rows via LABEL."
  if budget < 1: raise ValueError("choose: budget must be at least 1")
  tbl   = step2._load(blind_csv)
  blind = tbl.rows
  cap   = min(budget, len(blind))
  rng   = random.Random(ezr.the.Seed)
  todo  = rng.sample(range(len(blind)), len(blind))
  lab   = ezr.clone(tbl)               # learns the goal stats as we go
  goals = _goals_for(lab, goal)
  seen  = {}                           # row index -> full row, goals filled

  def ask(i: int) -> None:
    full = _fill(tbl, blind[i], label(i))
    seen[i] = full
    ezr.addRow(lab, full)

  for _ in range(min(START, cap)):
    ask(todo.pop())

  while len(seen) < cap:
    ranked = sorted(seen, key=lambda i: _cost(lab, seen[i], goals))
    k    = max(1, int(sqrt(len(ranked))))
    best = ezr.clone(tbl, [seen[i] for i in ranked[:k]])
    rest = ezr.clone(tbl, [seen[i] for i in ranked[k:]])
    cb, cr = ezr.mids(best), ezr.mids(rest)
    todo.sort(key=lambda i: ezr.xdist(tbl, blind[i], cr)
                          - ezr.xdist(tbl, blind[i], cb))
    ask(todo.pop())

  return min(seen, key=lambda i: _cost(lab, seen[i], goals))

