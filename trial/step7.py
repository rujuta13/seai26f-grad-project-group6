# step7.py
import math
import random
import ezr

ALIAS = {"speed": "performance"}   # client words -> goal base names

def _base(s: str) -> str:
  "A column name without its + or - sign, in lower case."
  return s.rstrip("+-").lower()

def _value(d: dict, name: str):
  "The label for column NAME, whether keyed with its sign or without."
  for k in (name, name.rstrip("+-")):
    if k in d: return d[k]
  return None

def _goal_ats(hdr: list[str], goal) -> list[int]:
  "Columns of the numeric goals in HDR, or only the one named GOAL."
  ats = [at for at, s in enumerate(hdr) if s[-1] in "+-"]
  if not ats: raise ValueError("choose: the file has no goal columns")
  if goal is None: return ats
  g    = str(goal).rstrip("+-").lower()
  g    = ALIAS.get(g, g)
  hits = [at for at in ats if _base(hdr[at]) == g]
  hits = hits or [at for at in ats if g and g in _base(hdr[at])]
  if not hits: raise ValueError(f"choose: no goal named {goal!r}")
  return hits

def choose(blind_csv, label, budget=30, goal=None) -> int:
  """Index of the best row in BLIND_CSV (goals all '?'), calling LABEL
  at most BUDGET times. If GOAL names one goal, optimise only that one."""
  if budget < 1: raise ValueError("choose: budget must be at least 1")
  raw   = [[str(s) for s in row] for row in ezr.csv(blind_csv)]
  hdr, body = raw[0], raw[1:]
  n     = len(body)
  if n == 0: raise ValueError("choose: no rows to choose from")
  goals = _goal_ats(hdr, goal)
  # unselected goals become ignored columns (names ending X), so ezr skips them
  names = [s.rstrip("+-") + "X"
           if (at not in goals and s[-1] in "+-") else s
           for at, s in enumerate(hdr)]
  has_x = any(s[-1] not in "X+-!" for s in names)   # ezr's input test
  calls = 0
  filled: dict[int, list] = {}

  def ask(i: int) -> dict:
    nonlocal calls
    if calls >= budget:
      raise RuntimeError(f"choose: more than {budget} labels requested")
    calls += 1
    return label(i) or {}

  def take(i: int) -> None:
    d   = ask(i)
    row = list(body[i])
    for at in goals:
      v = _value(d, hdr[at])
      if isinstance(v, str): v = ezr.atom(v)
      ok = isinstance(v, (int, float)) and not isinstance(v, bool)
      row[at] = v if ok else "?"
    filled[i] = row

  def complete(i: int) -> bool:
    return all(isinstance(filled[i][at], (int, float)) for at in goals)

  def table(ids: list[int]) -> ezr.TBL:
    return ezr.Tbl([names] + [filled[i] for i in ids])

  m     = min(budget, n)
  rng   = random.Random(ezr.the.Seed)
  order = list(range(n))
  rng.shuffle(order)
  for i in order[:min(ezr.the.Start, m)]: take(i)

  while len(filled) < m:
    done = [i for i in filled if complete(i)]
    pool = [i for i in order if i not in filled]
    if len(done) < 4 or not has_x:
      nxt = pool[0]                       # too little to model: explore
    else:
      tbl    = table(done)
      ys     = [ezr.ydist(tbl, filled[i]) for i in done]
      ranked = sorted(zip(ys, done))
      k      = max(1, int(math.sqrt(len(done))))
      best   = [i for _, i in ranked[:k]]
      rest   = [i for _, i in ranked[k:]]
      cb     = ezr.mids(ezr.clone(tbl, [filled[i] for i in best]))
      cr     = ezr.mids(ezr.clone(tbl, [filled[i] for i in rest]))
      nxt    = max(pool, key=lambda i: ezr.xdist(tbl, body[i], cr)
                                     - ezr.xdist(tbl, body[i], cb))
    take(nxt)

  done = [i for i in filled if complete(i)]
  if not done: return min(filled)
  tbl = table(done)
  ys  = [ezr.ydist(tbl, filled[i]) for i in done]
  return done[ys.index(min(ys))]

