# step24.py
"Step 24: flow, for the client. Say straight away which new customers are worth a welcome offer, and warn when this month's sign-ups stop looking like our usual customers."
import math
import os
import random
import ezr

QUANTILE = 0.25   # a customer is worth a welcome offer if their goals are in train's best 25%
WINDOW   = 15     # recent sign-ups compared against the train baseline, for drift
BASE_N   = 60     # train customers sampled to build the drift baseline
SEED     = 1234567891
SUFFX    = ("+", "-", "!")

_CACHE: dict = {}  # (path, mtime) -> (table, keep cutoff or None, drift baseline)

def _key(train_csv: str) -> tuple:
  "Identity of TRAIN_CSV: its absolute path and when it last changed."
  path = os.path.abspath(os.path.expanduser(train_csv))
  return (path, os.path.getmtime(path))

def _isnum(v) -> bool:
  "True for a usable finite number."
  return (isinstance(v, (int, float)) and not isinstance(v, bool)
          and math.isfinite(v))

def _coerce(v):
  "Strings become atoms; other values pass through."
  return ezr.atom(v) if isinstance(v, str) else v

def _full(tbl, row) -> list:
  """Lift ROW into a full-width row. Accepts a {column name: value} dict
  (goal suffixes optional), a full-width list, or a list with goal columns
  removed (header order). Any cell not given is '?'. Non-numeric cells in
  numeric columns become '?'."""
  names = tbl.names
  out   = ["?"] * len(names)
  if isinstance(row, dict):
    for at, n in enumerate(names):
      if n in row:
        out[at] = _coerce(row[n])
      elif n[-1:] in SUFFX and n[:-1] in row:
        out[at] = _coerce(row[n[:-1]])
  else:
    row = list(row)
    if len(row) == len(names):
      out = [_coerce(v) for v in row]
    else:
      keep = [i for i, n in enumerate(names) if n[-1:] not in SUFFX]
      for i, v in zip(keep, row): out[i] = _coerce(v)
  for c in tbl.cols.values():
    if type(c) is ezr.Num and not _isnum(out[c.at]):
      out[c.at] = "?"
  return out

def _ydist(tbl, row):
  "Distance of ROW's numeric goals from their best; None if no goal is known."
  sq, n = 0.0, 0
  for c in tbl.y:
    if type(c) is not ezr.Num: continue
    v = row[c.at]
    if not _isnum(v): continue
    sq += (ezr.norm(c, v) - c.goal) ** 2
    n  += 1
  return (sq / n) ** .5 if n else None

def _nn(tbl, row, skip: int | None = None) -> float:
  "Distance from ROW to its nearest train row over the x columns (ignoring train row SKIP)."
  return min((ezr.xdist(tbl, row, t)
              for j, t in enumerate(tbl.rows) if j != skip),
             default=1.0)

def _load(train_csv: str):
  "Train table, keep cutoff (None if no train goals), and drift baseline; cached per file version."
  key = _key(train_csv)
  if key not in _CACHE:
    tbl  = ezr.Tbl(ezr.csv(train_csv))
    ys   = sorted(d for d in (_ydist(tbl, r) for r in tbl.rows)
                  if d is not None)
    thr  = ys[int(QUANTILE * (len(ys) - 1))] if ys else None
    n    = len(tbl.rows)
    rng  = random.Random(SEED)
    pick = rng.sample(range(n), min(BASE_N, n))
    base = [_nn(tbl, tbl.rows[i], skip=i) for i in pick]
    _CACHE[key] = (tbl, thr, base)
  return _CACHE[key]

def stream(train_csv, rows) -> list[bool]:
  """One keep decision per row, in order. A row is kept (worth a welcome
  offer) if its goals are among the best QUANTILE of the train rows. Each
  decision uses only TRAIN_CSV and the row at hand."""
  tbl, thr, _ = _load(train_csv)
  out = []
  for r in rows:
    d = _ydist(tbl, _full(tbl, r))
    out.append(thr is not None and d is not None and d <= thr)
  return out

def drift(train_csv, rows) -> int | None:
  """Index into ROWS of the first row at which the last WINDOW rows no
  longer look like train (by nearest-train distance over the x columns),
  or None if they never do. Uses only TRAIN_CSV and rows up to that index."""
  tbl, _, base = _load(train_csv)
  if not base: return None
  recent: list[float] = []
  for i, r in enumerate(rows):
    recent.append(_nn(tbl, _full(tbl, r)))
    if len(recent) > WINDOW: recent.pop(0)
    if len(recent) == WINDOW and not ezr.same(base, recent):
      return i
  return None

