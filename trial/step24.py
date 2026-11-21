# step24.py
import math
import random
import statistics
import ezr

GOOD = 0.25   # keep a sign-up if its predicted goal score is in the best quarter of training
K_MIN = 5     # fewest nearest training rows that count as evidence
WIN = 10      # drift: new rows are judged in blocks of this many
BASE = 100    # drift: training rows used to measure normal novelty

_trains: dict[str, ezr.TBL] = {}      # load each training file once
_knowns: dict[str, ezr.ROWS] = {}     # cleaned training rows, once per file
_cuts: dict[str, float] = {}          # keep cutoff on ydist, once per file
_bases: dict[str, list[float]] = {}   # normal novelty, once per file

def _load(train_csv: str) -> ezr.TBL:
  "Load TRAIN_CSV, and remember it for later calls."
  if train_csv not in _trains:
    _trains[train_csv] = ezr.Tbl(ezr.csv(train_csv))
  return _trains[train_csv]

def _is_num(v) -> bool:
  "True if V is a real number (a bool is not)."
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def _cell(v):
  "A raw cell as an ezr atom; None, blank and '?' become '?'."
  if v is None: return "?"
  if isinstance(v, str):
    s = v.strip()
    return "?" if s in ("", "?") else ezr.atom(s)
  return v

def _align(tbl: ezr.TBL, row) -> ezr.ROW:
  """Align ROW to TBL's columns. ROW is a dict by column name (with or
  without its + or - sign), or a list over every column or over the
  non-goal columns. Missing is '?'."""
  q = ["?"] * len(tbl.names)
  if isinstance(row, dict):
    pairs = [(at, row.get(name, row.get(name.rstrip("+-"))))
             for at, name in enumerate(tbl.names)]
  elif len(row) == len(tbl.names):
    pairs = list(enumerate(row))
  else:
    keep  = [at for at, n in enumerate(tbl.names) if n[-1] not in "+-"]
    pairs = list(zip(keep, row))
  for at, v in pairs: q[at] = _cell(v)
  return q

def _fit(tbl: ezr.TBL, q: ezr.ROW) -> ezr.ROW:
  "Q with any non-number in a number column blanked to '?'."
  out = list(q)
  for c in tbl.cols.values():
    if type(c) is ezr.Num and not _is_num(out[c.at]): out[c.at] = "?"
  return out

def _goals(tbl: ezr.TBL) -> list:
  "The numeric goal columns of TBL (names ending + or -)."
  return [c for c in tbl.y if type(c) is ezr.Num]

def _complete(goals: list, r: ezr.ROW) -> bool:
  "True if every goal cell of R is a number."
  return all(_is_num(r[c.at]) for c in goals)

def _ydist(goals: list, r: ezr.ROW) -> float:
  "How far R's goals are from the best they could be (0 is best)."
  if not goals: return 0.0
  return ezr.dist((abs(ezr.norm(c, r[c.at]) - c.goal) for c in goals),
                  len(goals))

def _known(train_csv: str) -> ezr.ROWS:
  "Training rows, cleaned to match the table's columns."
  if train_csv not in _knowns:
    tbl = _load(train_csv)
    _knowns[train_csv] = [_fit(tbl, r) for r in tbl.rows]
  return _knowns[train_csv]

def _cut(train_csv: str) -> float:
  "The ydist at or below which a training-like row counts as good: best GOOD of training."
  if train_csv not in _cuts:
    tbl   = _load(train_csv)
    goals = _goals(tbl)
    ys    = sorted(_ydist(goals, r) for r in _known(train_csv)
                   if goals and _complete(goals, r))
    _cuts[train_csv] = (ys[min(len(ys) - 1, int(GOOD * (len(ys) - 1)))]
                        if ys else float("-inf"))
  return _cuts[train_csv]

def _nearest(tbl: ezr.TBL, known: ezr.ROWS, q: ezr.ROW) -> ezr.ROWS:
  "The training rows nearest to Q over the input columns."
  k = min(len(known), max(K_MIN, int(math.sqrt(len(known)))))
  return sorted(known, key=lambda r: ezr.xdist(tbl, q, r))[:k]

def _keep(tbl: ezr.TBL, goals: list, known: ezr.ROWS, cut: float,
          q: ezr.ROW) -> bool:
  "Keep Q if the goals of its nearest training rows are, typically, in the best GOOD."
  if not goals or not tbl.x: return False
  if not any(q[c.at] != "?" for c in tbl.x): return False  # no inputs: cannot judge
  near = _nearest(tbl, known, q)
  ys   = [_ydist(goals, r) for r in near if _complete(goals, r)]
  if len(ys) < K_MIN: return False
  return statistics.median(ys) <= cut

def stream(train_csv, rows) -> list[bool]:
  """Keep a row (True) if the training rows most like its inputs were good on
  the goals. Uses only TRAIN_CSV and the inputs of each row; goal cells in the
  row are never used, since a new sign-up's goals are not yet known."""
  tbl   = _load(train_csv)
  goals = _goals(tbl)
  known = _known(train_csv)
  cut   = _cut(train_csv)
  out: list[bool] = []
  for row in rows:
    q = _fit(tbl, _align(tbl, row))
    out.append(bool(_keep(tbl, goals, known, cut, q)))
  return out

def _baseline(train_csv: str) -> list[float]:
  "How novel training rows look to the rest of training: nearest-neighbour xdist."
  if train_csv not in _bases:
    tbl   = _load(train_csv)
    known = _known(train_csv)
    if not tbl.x or len(known) < 2:
      _bases[train_csv] = []
    else:
      k   = min(BASE, len(known))
      idx = random.Random(ezr.the.Seed).sample(range(len(known)), k)
      _bases[train_csv] = [
        min(ezr.xdist(tbl, known[i], s)
            for j, s in enumerate(known) if j != i)
        for i in idx]
  return _bases[train_csv]

def drift(train_csv, rows) -> int | None:
  """Index (at the end of a block of WIN new rows) where the new rows are
  more novel than usual, and not the same distribution as training's normal
  novelty. Uses only TRAIN_CSV and the rows seen so far; goal cells are ignored."""
  tbl  = _load(train_csv)
  base = _baseline(train_csv)
  if not base: return None
  base_mid = statistics.median(base)
  known    = _known(train_csv)
  recent: list[float] = []
  for i, row in enumerate(rows):
    q = _fit(tbl, _align(tbl, row))
    recent.append(min((ezr.xdist(tbl, q, s) for s in known), default=0.0))
    if len(recent) % WIN == 0:
      block = recent[-WIN:]
      if (statistics.median(block) > base_mid
          and not ezr.same(base, block)):
        return i
  return None

