# step7.py
"Step 7: choose. Find the best build while benchmarking at most BUDGET builds."
import math
import random
import ezr

START = 4      # builds labelled at random before any guided choice

def _val(v):
  "Strings become atoms; other values pass through."
  return ezr.atom(v) if isinstance(v, str) else v

def _name(s: str) -> str:
  "A goal name without its +, - or ! suffix."
  return s[:-1] if s[-1:] in "+-!" else s

def _goals(tbl, goal) -> list:
  "Numeric goal columns to optimise: all of them, or the one named GOAL."
  ys = list(tbl.y)
  if goal is not None:
    want  = goal.lower()
    tests = (lambda c: c.txt == goal,
             lambda c: c.txt.lower() == want,
             lambda c: _name(c.txt).lower() == _name(want),
             lambda c: _name(c.txt).lower().startswith(_name(want)))
    hits = []
    for test in tests:
      hits = [c for c in ys if test(c)]
      if hits: break
    if len(hits) != 1:
      raise ValueError(f"choose: goal {goal!r} must name exactly one goal")
    ys = hits
  ys = [c for c in ys if type(c) is ezr.Num]
  if not ys: raise ValueError("choose: no numeric goal to optimise")
  return ys

def _scores(goals, known: dict) -> dict:
  "Each labelled build's distance from the ideal, over the chosen goals only."
  norms = {}
  for c in goals:
    num = ezr.Num(c.txt, c.at)
    for vals in known.values():
      v = vals.get(c.txt)
      if isinstance(v, (int, float)): ezr.add(num, v)
    norms[c.txt] = num
  out = {}
  for i, vals in known.items():
    sq, n = 0.0, 0
    for c in goals:
      v = vals.get(c.txt)
      if not isinstance(v, (int, float)): continue
      sq += (ezr.norm(norms[c.txt], v) - c.goal) ** 2
      n  += 1
    out[i] = (sq / n) ** .5 if n else math.inf
  return out

def choose(blind_csv, label, budget=30, goal=None) -> int:
  "Index of the best build found, after labelling at most BUDGET builds."
  tbl  = ezr.Tbl(ezr.csv(blind_csv))
  rows = tbl.rows
  if not rows: raise ValueError("choose: no builds to choose from")
  if budget < 1: raise ValueError("choose: budget must be at least 1")
  goals  = _goals(tbl, goal)
  budget = min(budget, len(rows))
  known: dict[int, dict] = {}
  near:  dict[int, list[float]] = {}
  spent  = 0

  def ask(i: int) -> None:
    nonlocal spent
    if i in known: return
    if spent >= budget: raise RuntimeError("choose: budget exhausted")
    spent += 1
    known[i] = {k: _val(v) for k, v in label(i).items()}

  def gaps(b: int) -> list[float]:
    "Input distance from labelled build B to every build (cached)."
    if b not in near:
      near[b] = [ezr.xdist(tbl, z, rows[b]) for z in rows]
    return near[b]

  def lure(z: int, best: list, rest: list) -> float:
    "Closer to the best labelled builds than to the rest: lower is better."
    nb = min(gaps(b)[z] for b in best)
    return nb - min(gaps(b)[z] for b in rest) if rest else nb

  order = random.Random(ezr.the.Seed).sample(range(len(rows)), len(rows))
  for i in order[:min(START, budget)]: ask(i)

  while spent < budget:
    scores = _scores(goals, known)
    ranked = sorted(known, key=lambda i: (scores[i], i))
    k      = max(1, int(math.sqrt(len(ranked))))
    best, rest = ranked[:k], ranked[k:]
    pick = min((z for z in range(len(rows)) if z not in known),
               key=lambda z: (lure(z, best, rest), z))
    ask(pick)

  scores = _scores(goals, known)
  return min(known, key=lambda i: (scores[i], i))

