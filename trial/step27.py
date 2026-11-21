# step27.py
"""Step 27: choose.  Find the best customers to target while test-mailing
only a small budget of them, and optionally optimise only one goal.

The client's two requests map onto the arguments:
  - "best customers, test-mailing only 50 people": budget=50.
  - "store visits rather than web orders": goal="NUMSTOREPURCHASES+".

Identifier and text columns (ID, num, Rank, Name, Nation, League, Team, url)
are left out, as in steps 11 and 21, and so are columns whose names end in
"X".  Numeric columns keep only numbers; anything else counts as unknown.

Uses only the standard library plus ezr.
"""
import math
import random
from collections.abc import Callable
import ezr

_DROP = {"ID", "num", "Rank", "Name", "Nation", "League", "Team", "url"}


def _num(v) -> bool:
  "True if V is a usable finite number (not text, not missing, not a bool)."
  if isinstance(v, bool) or not isinstance(v, (int, float)): return False
  return not isinstance(v, float) or math.isfinite(v)


def _clean(v):
  "A cell as ezr reads it: strings are cast; blanks, '?' and non-finite numbers are unknown."
  if v is None: return "?"
  if isinstance(v, str):
    s = v.strip()
    if s in ("", "?"): return "?"
    v = ezr.atom(s)
  if isinstance(v, float) and not math.isfinite(v): return "?"
  return v


def _stem(name) -> str:
  "A column name without its goal sign, lower-cased: 'Cpu-' -> 'cpu'."
  return str(name).strip().rstrip("+-!").lower()


def _load(blind_csv: str) -> ezr.TBL:
  "Read BLIND_CSV into an ezr table: identifier and text columns dropped, numeric columns kept numeric."
  raw = list(ezr.csv(blind_csv))
  if not raw: raise ValueError("choose: blind_csv has no rows")
  names = [str(h).strip() for h in raw[0]]
  kept  = [at for at, nm in enumerate(names)
           if nm and nm not in _DROP and not nm.endswith("X")]
  head  = [names[at] for at in kept]
  numeric = [h[:1].isupper() for h in head]    # ezr types a column the same way
  rows = []
  for r in raw[1:]:
    r = [_clean(v) for v in r][:len(names)]
    r += ["?"] * (len(names) - len(r))
    row = [r[at] for at in kept]
    rows.append([v if (not numeric[j] or _num(v)) else "?"
                 for j, v in enumerate(row)])
  return ezr.Tbl([head] + rows)


def _goals(tbl: ezr.TBL, goal: str | None) -> list[ezr.COL]:
  "The numeric goal columns to optimise: all of them, or the one named GOAL."
  nums = [c for c in tbl.y if isinstance(c, ezr.Num)]
  if not nums:
    raise ValueError("choose: blind_csv has no numeric goal columns")
  if goal is None: return nums
  hits = [c for c in nums if _stem(c.txt) == _stem(goal)]
  if not hits:
    names = ", ".join(c.txt for c in nums)
    raise ValueError(f"choose: no goal named {goal!r} (goals are: {names})")
  return hits


def choose(blind_csv: str, label: Callable[[int], dict],
           budget: int = 30, goal: str | None = None) -> int:
  """Index of the best row found.  BLIND_CSV has every goal cell '?'.
  LABEL(i) returns row i's goals as a dict, and is called at most BUDGET
  times (a further call raises RuntimeError).  If GOAL names one goal, only
  that goal is optimised; GOAL may be given with or without its sign."""
  tbl  = _load(blind_csv)
  rows = tbl.rows
  if not rows: raise ValueError("choose: blind_csv has no rows")
  if budget < 1: raise ValueError("choose: budget must be at least 1")
  active = _goals(tbl, goal)
  ats    = [c.at for c in active]
  n      = len(rows)
  rng    = random.Random(ezr.the.Seed)     # local rng: no global side effects
  done: dict[int, ezr.ROW] = {}            # row index -> row with goals known
  dists: dict[int, list[float]] = {}       # labelled row -> xdist to every row

  def xd(z: int, j: int) -> float:
    "How far row Z is from row J over the x columns (0 if there are none)."
    return ezr.xdist(tbl, rows[z], rows[j]) if tbl.x else 0.0

  def ask(i: int) -> None:
    "Label row I, once, and remember its goals and its distances."
    if len(done) >= budget:
      raise RuntimeError(f"choose: budget of {budget} labels exhausted")
    got   = dict(label(i) or {})
    stems = {_stem(k): v for k, v in got.items()}
    row   = list(rows[i])
    for c in tbl.y:
      if not isinstance(c, ezr.Num): continue
      v = got[c.txt] if c.txt in got else stems.get(_stem(c.txt), "?")
      v = _clean(v)
      row[c.at] = v if _num(v) else "?"
    done[i]  = row
    dists[i] = [xd(z, i) for z in range(n)]

  def score(lab: ezr.TBL, row: ezr.ROW) -> float:
    "Distance from the ideal on the chosen goals; lower is better, inf if unknown."
    gaps = []
    for at in ats:
      c, v = lab.cols[at], row[at]
      if v == "?" or isinstance(v, str): return math.inf
      gaps.append(abs(ezr.norm(c, v) - c.goal))
    return ezr.dist(gaps, len(gaps))

  def scores() -> dict[int, float]:
    "Score of every labelled row, normalised by the labelled rows so far."
    ids = sorted(done)
    lab = ezr.clone(tbl, [done[i] for i in ids])
    return {i: score(lab, done[i]) for i in ids}

  def predict(z: int, sc: dict[int, float]) -> float:
    "Mean score of the nearest labelled rows to unlabelled row Z."
    k    = max(1, int(math.sqrt(len(sc))))
    near = sorted(sc, key=lambda j: (dists[j][z], j))[:k]
    return sum(sc[j] for j in near) / len(near)

  # Start with a few random labels, then label the predicted best rows,
  # with a random row now and then so the search does not stall.
  start = min(budget, n, max(4, budget // 4))
  for i in rng.sample(range(n), start): ask(i)
  step = 0
  while len(done) < min(budget, n):
    step += 1
    todo = [z for z in range(n) if z not in done]
    if step % 4 == 0:
      ask(rng.choice(todo))
      continue
    sc = scores()
    ask(min(todo, key=lambda z: (predict(z, sc), z)))

  sc = scores()
  return min(sc, key=lambda i: (sc[i], i))

