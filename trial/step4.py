# step4.py
import ezr

TOP  = 0.10   # stream: keep rows in the best 10% of training ydist
BASE = 100    # drift: training rows used to measure normal novelty
WIN  = 10     # drift: recent rows compared against normal

_trains: dict[str, ezr.TBL] = {}    # load each training file once
_bases: dict[str, list[float]] = {} # normal novelty, once per file

def _load(train_csv: str) -> ezr.TBL:
  "Load TRAIN_CSV, and remember it for later calls."
  if train_csv not in _trains:
    _trains[train_csv] = ezr.Tbl(ezr.csv(train_csv))
  return _trains[train_csv]

def _goals(tbl: ezr.TBL) -> list:
  "The numeric goal columns of TBL."
  return [c for c in tbl.y if type(c) is ezr.Num]

def _align(tbl: ezr.TBL, row) -> ezr.ROW:
  "Align ROW (a dict by name, or a list by position) to TBL's columns. Missing is '?'."
  q = ["?"] * len(tbl.names)
  if isinstance(row, dict):
    pairs = [(at, row.get(name)) for at, name in enumerate(tbl.names)]
  else:
    pairs = list(enumerate(row))[:len(q)]
  for at, v in pairs:
    if v is None or (isinstance(v, str) and v.strip() == ""): continue
    q[at] = ezr.atom(v) if isinstance(v, str) else v
  return q

def _complete(goals: list, q: ezr.ROW) -> bool:
  "True if every goal cell of Q is a number."
  return all(isinstance(q[c.at], (int, float)) for c in goals)

def _ydist(goals: list, q: ezr.ROW) -> float:
  "How far Q's goals are from the best they could be (0 is best)."
  if not goals: return 0.0
  return ezr.dist((abs(ezr.norm(c, q[c.at]) - c.goal) for c in goals),
                  len(goals))

def _xdist(tbl: ezr.TBL, a: ezr.ROW, b: ezr.ROW) -> float:
  "Distance over the x columns; 0 if there are none."
  return ezr.xdist(tbl, a, b) if tbl.x else 0.0

def stream(train_csv, rows) -> list[bool]:
  "Keep a row if its goals are in the best TOP of TRAIN_CSV's goals."
  tbl   = _load(train_csv)
  goals = _goals(tbl)
  ys    = sorted(_ydist(goals, r) for r in tbl.rows
                 if goals and _complete(goals, r))
  cut   = ys[min(len(ys) - 1, int(TOP * len(ys)))] if ys else -1.0
  out: list[bool] = []
  for row in rows:
    q = _align(tbl, row)
    keep = bool(goals) and _complete(goals, q) and _ydist(goals, q) <= cut
    out.append(bool(keep))
  return out

def _baseline(train_csv: str) -> list[float]:
  "How novel training rows look to the rest of training: nearest-neighbour xdist."
  if train_csv not in _bases:
    tbl  = _load(train_csv)
    rows = tbl.rows
    if len(rows) < 2:
      _bases[train_csv] = []
    else:
      k = min(BASE, len(rows))
      _bases[train_csv] = [
        min(_xdist(tbl, rows[i], s) for j, s in enumerate(rows) if j != i)
        for i in range(k)]
  return _bases[train_csv]

def drift(train_csv, rows) -> int | None:
  "First index where the last WIN rows look unlike training rows, by novelty."
  tbl  = _load(train_csv)
  base = _baseline(train_csv)
  if not base: return None
  recent: list[float] = []
  for i, row in enumerate(rows):
    q = _align(tbl, row)
    recent.append(min((_xdist(tbl, q, s) for s in tbl.rows), default=0.0))
    if len(recent) >= WIN and not ezr.same(base, recent[-WIN:]):
      return i
  return None

