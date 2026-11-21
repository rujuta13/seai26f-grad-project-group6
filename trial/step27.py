# step27.py
"Step 27: choose. Find the best customers to target while test-mailing at most BUDGET of them, optionally optimising only the goal named GOAL (e.g. GOAL = 'NUMSTOREPURCHASES' for store visits)."
import math
import random
import ezr

START = 4            # customers labelled at random before any guided choice
SEED  = 1234567891   # fixed seed, so results repeat

def choose(blind_csv, label, budget=30, goal=None) -> int:
  """Index of the best row found. BLIND_CSV has every goal cell '?'.
  LABEL(i) returns row I's goals as a dict (goal name -> value) and may be
  called at most BUDGET times; calling it more raises RuntimeError. If GOAL
  names one goal, only that goal is optimised."""
  tbl  = ezr.Tbl(ezr.csv(blind_csv))
  rows = tbl.rows
  if not rows: raise ValueError("choose: no rows to choose from")
  if (not isinstance(budget, int) or isinstance(budget, bool)
      or budget < 1):
    raise ValueError("choose: budget must be a whole number of at least 1")

  def isnum(v) -> bool:
    "True for a usable finite number."
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))

  def atom(v):
    "Strings become atoms; other values pass through."
    return ezr.atom(v) if isinstance(v, str) else v

  def name(s: str) -> str:
    "A goal name without its +, - or ! suffix."
    return s[:-1] if s[-1:] in ("+", "-", "!") else s

  # which goals to optimise
  ys = [c for c in tbl.y if type(c) is ezr.Num]
  if goal is not None:
    want  = str(goal).strip()
    tests = (lambda c: c.txt == want,
             lambda c: c.txt.lower() == want.lower(),
             lambda c: name(c.txt).lower() == name(want).lower(),
             lambda c: name(c.txt).lower().startswith(name(want).lower()))
    hits = []
    for test in tests:
      hits = [c for c in tbl.y if test(c)]
      if hits: break
    if len(hits) != 1:
      raise ValueError(f"choose: goal {goal!r} must name exactly one goal")
    ys = [c for c in hits if type(c) is ezr.Num]
    if not ys: raise ValueError(f"choose: goal {goal!r} is not numeric")
  if not ys: raise ValueError("choose: no numeric goal to optimise")

  # input columns that help distance: numbers, and symbols that repeat
  # (a symbol where every value is distinct is an identifier, like a url)
  cols = [c for c in tbl.x if type(c) is ezr.Num or len(c.has) < c.n]

  def gap(c, a, b) -> float:
    "Distance between two cells of column C; unknown is far."
    if type(c) is ezr.Num:
      if not (isnum(a) and isnum(b)): return 1.0
      return abs(ezr.norm(c, a) - ezr.norm(c, b))
    if a == "?" or b == "?": return 1.0
    return 0.0 if a == b else 1.0

  near: dict[int, list[float]] = {}

  def gaps(b: int) -> list[float]:
    "Input distance from row B to every row (cached)."
    if b not in near:
      near[b] = [math.sqrt(sum(gap(c, r[c.at], rows[b][c.at]) ** 2
                               for c in cols) / max(1, len(cols)))
                 for r in rows]
    return near[b]

  known: dict[int, dict] = {}
  spent = 0

  def ask(i: int) -> None:
    "Label row I, if not yet labelled, within the budget."
    nonlocal spent
    if i in known: return
    if spent >= budget: raise RuntimeError("choose: budget exhausted")
    spent += 1
    vals = label(i)
    known[i] = {c.txt: atom(vals.get(c.txt, "?")) for c in ys}

  def scores() -> dict[int, float]:
    "Each labelled row's distance from the ideal, over the chosen goals."
    norms = {}
    for c in ys:
      num = ezr.Num(c.txt, c.at)
      for vals in known.values():
        v = vals[c.txt]
        if isnum(v): ezr.add(num, v)
      norms[c.txt] = num
    out = {}
    for i, vals in known.items():
      sq, n = 0.0, 0
      for c in ys:
        v = vals[c.txt]
        if not isnum(v): continue
        sq += (ezr.norm(norms[c.txt], v) - c.goal) ** 2
        n  += 1
      out[i] = math.sqrt(sq / n) if n else math.inf
    return out

  def lure(z: int, best: list, rest: list) -> float:
    "Closer to the best labelled rows than to the rest: lower is better."
    nb = min(gaps(b)[z] for b in best)
    return nb - min(gaps(b)[z] for b in rest) if rest else nb

  budget = min(budget, len(rows))
  order  = random.Random(SEED).sample(range(len(rows)), len(rows))
  for i in order[:min(START, budget)]: ask(i)

  while spent < budget:
    sc     = scores()
    ranked = sorted(known, key=lambda i: (sc[i], i))
    k      = max(1, int(math.sqrt(len(ranked))))
    best, rest = ranked[:k], ranked[k:]
    pick = min((z for z in range(len(rows)) if z not in known),
               key=lambda z: (lure(z, best, rest), z))
    ask(pick)

  sc = scores()
  return min(known, key=lambda i: (sc[i], i))

