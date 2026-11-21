# step7.py
"""Step 7: choose.  Find the best flag set while benchmarking at most BUDGET
builds, and optionally optimise only one goal.

Uses only the standard library plus ezr.
"""
import math
import random
from collections.abc import Callable
import ezr


def _load(blind_csv: str) -> ezr.TBL:
  "Read BLIND_CSV into an ezr table, keeping file order."
  return ezr.Tbl(ezr.csv(blind_csv))


def _key(name: str) -> str:
  "A goal name without its direction sign, lower-cased: 'Cpu-' -> 'cpu'."
  return name.rstrip("+-!").lower()


def _goals(tbl: ezr.TBL, goal: str | None) -> list[ezr.COL]:
  "The goal columns to optimise: all of them, or the one named GOAL."
  if not tbl.y:
    raise ValueError("choose: blind_csv has no goal columns")
  if goal is None: return list(tbl.y)
  hits = [c for c in tbl.y if _key(c.txt) == _key(goal)]
  if not hits:
    names = ", ".join(c.txt for c in tbl.y)
    raise ValueError(f"choose: no goal named {goal!r} (goals are: {names})")
  return hits


def choose(blind_csv: str, label: Callable[[int], dict],
           budget: int = 30, goal: str | None = None) -> int:
  """Index of the best row found.  BLIND_CSV has every goal cell '?'.
  LABEL(i) returns row i's goals as a dict, and is called at most BUDGET
  times.  If GOAL names one goal, only that goal is optimised."""
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
    got = label(i)
    row = list(rows[i])
    for c in tbl.y:
      if c.txt in got:
        v = got[c.txt]
        v = ezr.atom(v) if isinstance(v, str) else v
        row[c.at] = v if isinstance(v, (int, float)) else "?"
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

