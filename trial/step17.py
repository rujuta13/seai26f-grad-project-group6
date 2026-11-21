# step17.py
import math
import random
import ezr

def choose(blind_csv, label, budget=30, goal=None) -> int:
  """Index of the best row in BLIND_CSV (goals all '?'), calling LABEL
  at most BUDGET times. If GOAL names one goal, optimise only that one."""
  if budget < 1: raise ValueError("choose: budget must be at least 1")
  raw = list(ezr.csv(blind_csv))
  if len(raw) < 2: raise ValueError("choose: no rows to choose from")
  hdr, body = [str(s) for s in raw[0]], raw[1:]
  n = len(body)

  # which columns are goals (+ or -), and which of those we optimise
  ats = [at for at, s in enumerate(hdr) if s[-1] in "+-"]
  if not ats: raise ValueError("choose: the file has no goal columns")
  if goal is None:
    goals = ats
  else:
    g     = str(goal).rstrip("+-").lower()
    goals = [at for at in ats if hdr[at].rstrip("+-").lower() == g]
    if not goals: raise ValueError(f"choose: no goal named {goal!r}")

  # unselected goals and klass columns become ignored (names ending X)
  names = []
  for at, s in enumerate(hdr):
    if at in goals or s[-1] not in "+-!":
      names.append(s)
    else:
      names.append(s.rstrip("+-!") + "X")
  has_x = any(s[-1] not in "X+-!" for s in names)

  calls  = 0
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
      v = None
      for key in (hdr[at], hdr[at].rstrip("+-")):
        if key in d:
          v = d[key]
          break
      if isinstance(v, str): v = ezr.atom(v)
      ok = isinstance(v, (int, float)) and not isinstance(v, bool)
      row[at] = v if ok else "?"
    filled[i] = row

  def complete(i: int) -> bool:
    return all(isinstance(filled[i][at], (int, float))
               and not isinstance(filled[i][at], bool) for at in goals)

  def table(ids: list[int]) -> ezr.TBL:
    return ezr.Tbl([names] + [filled[i] for i in ids])

  m     = min(budget, n)
  rng   = random.Random(ezr.the.Seed)
  order = list(range(n))
  rng.shuffle(order)
  for i in order[:min(ezr.the.Start, m)]: take(i)

  while len(filled) < m:
    pool = [i for i in order if i not in filled]
    done = [i for i in filled if complete(i)]
    if len(done) < 4 or not has_x:
      nxt = pool[0]                       # too little to model: explore
    else:
      tbl    = table(done)
      ranked = sorted((ezr.ydist(tbl, filled[i]), i) for i in done)
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
  tbl    = table(done)
  ranked = sorted((ezr.ydist(tbl, filled[i]), i) for i in done)
  return ranked[0][1]

