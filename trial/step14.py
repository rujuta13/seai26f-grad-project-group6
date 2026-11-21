# step14.py
import statistics
import ezr

TOP  = 0.10   # stream: keep rows whose goals are in the best 10% of training
BASE = 100    # drift: training rows used to measure normal novelty
WIN  = 10     # drift: non-overlapping blocks of this many new rows

_trains: dict[str, ezr.TBL] = {}          # load each training file once
_knowns: dict[str, ezr.ROWS] = {}         # cleaned training rows, once per file
_cuts:   dict[str, float] = {}            # keep cutoff on ydist, once per file
_bases:  dict[str, list[float]] = {}      # normal novelty, once per file

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

def _complete(goals: list, q: ezr.ROW) -> bool:
  "True if every goal cell of Q is a number."
  return all(_is_num(q[c.at]) for c in goals)

def _ydist(goals: list, q: ezr.ROW) -> float:
  "How far Q's goals are from the best they could be (0 is best)."
  return ezr.dist((abs(ezr.norm(c, q[c.at]) - c.goal) for c in goals),
                  len(goals))

def _known(train_csv: str) -> ezr.ROWS:
  "Training rows, cleaned to match the table's columns."
  if train_csv not in _knowns:
    tbl = _load(train_csv)
    _knowns[train_csv] = [_fit(tbl, r) for r in tbl.rows]
  return _knowns[train_csv]

def _cut(train_csv: str) -> float:
  "The ydist below which a row counts as good: best TOP of training rows."
  if train_csv not in _cuts:
    tbl   = _load(train_csv)
    goals = _goals(tbl)
    ys    = sorted(_ydist(goals, r) for r in _known(train_csv)
                   if goals and _complete(goals, r))
    _cuts[train_csv] = (ys[min(len(ys) - 1, int(TOP * len(ys)))]
                        if ys else float("-inf"))
  return _cuts[train_csv]

def stream(train_csv, rows) -> list[bool]:
  "Keep a row if its goals are in the best TOP of TRAIN_CSV's goals."
  tbl   = _load(train_csv)
  goals = _goals(tbl)
  cut   = _cut(train_csv)
  out: list[bool] = []
  for row in rows:
    q    = _fit(tbl, _align(tbl, row))
    keep = bool(goals) and _complete(goals, q) and _ydist(goals, q) <= cut
    out.append(bool(keep))
  return out

def _baseline(train_csv: str) -> list[float]:
  "How novel training rows look to the rest of training: nearest-neighbour xdist."
  if train_csv not in _bases:
    tbl   = _load(train_csv)
    known = _known(train_csv)
    if not tbl.x or len(known) < 2:
      _bases[train_csv] = []
    else:
      k = min(BASE, len(known))
      _bases[train_csv] = [
        min(ezr.xdist(tbl, known[i], s)
            for j, s in enumerate(known) if j != i)
        for i in range(k)]
  return _bases[train_csv]

def drift(train_csv, rows) -> int | None:
  """First index (at the end of a block of WIN new rows) where those rows
  are less like training than usual: both more novel, and not the same
  distribution. Uses only training and the rows seen so far."""
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

