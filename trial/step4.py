# step4.py
"Step 4: flow. Keep good rows as they stream in; warn when they drift from train."
import random
import ezr

QUANTILE = 0.25   # a row is "worth keeping" if its goal distance is in train's best 25%
WINDOW   = 15     # recent rows compared against the train baseline, for drift
BASE_N   = 60     # train rows sampled to build the drift baseline
SEED     = 1234567891

_CACHE: dict = {}

def _coerce(v):
  "Strings become atoms; other values pass through."
  return ezr.atom(v) if isinstance(v, str) else v

def _full(tbl, row) -> list:
  """Lift ROW into a full-width row. Accepts a {column name: value} dict,
  a full-width list, or a list with goal columns removed (header order).
  Any cell not given is '?'."""
  names = tbl.names
  if isinstance(row, dict):
    out = ["?"] * len(names)
    for at, n in enumerate(names):
      if n in row: out[at] = _coerce(row[n])
    return out
  row = list(row)
  if len(row) == len(names):
    return [_coerce(v) for v in row]
  out  = ["?"] * len(names)
  keep = [i for i, n in enumerate(names) if n[-1] not in "+-!"]
  for i, v in zip(keep, row): out[i] = _coerce(v)
  return out

def _ydist(tbl, row):
  "Distance of ROW's numeric goals from their best; None if no goal is known."
  sq, n = 0.0, 0
  for c in tbl.y:
    if type(c) is not ezr.Num: continue
    v = row[c.at]
    if v == "?": continue
    sq += (ezr.norm(c, v) - c.goal) ** 2
    n  += 1
  return (sq / n) ** .5 if n else None

def _nn(tbl, row, skip: int | None = None) -> float:
  "Distance from ROW to its nearest train row (ignoring train row SKIP)."
  return min((ezr.xdist(tbl, row, t)
              for j, t in enumerate(tbl.rows) if j != skip),
             default=1.0)

def _load(train_csv: str):
  "Train table, keep-threshold, and drift baseline; cached per file."
  if train_csv not in _CACHE:
    tbl = ezr.Tbl(ezr.csv(train_csv))
    ys  = sorted(d for d in (_ydist(tbl, r) for r in tbl.rows)
                 if d is not None)
    thr = ys[int(QUANTILE * (len(ys) - 1))] if ys else float("inf")
    rng  = random.Random(SEED)
    pick = rng.sample(range(len(tbl.rows)), min(BASE_N, len(tbl.rows)))
    base = [_nn(tbl, tbl.rows[i], skip=i) for i in pick]
    _CACHE[train_csv] = (tbl, thr, base)
  return _CACHE[train_csv]

def stream(train_csv, rows) -> list[bool]:
  """One keep decision per row, in order. A row is kept if its goals are
  among the best QUANTILE of the train rows. Decisions use only TRAIN_CSV
  and the row at hand."""
  tbl, thr, _ = _load(train_csv)
  out = []
  for r in rows:
    d = _ydist(tbl, _full(tbl, r))
    out.append(d is not None and d <= thr)
  return out

def drift(train_csv, rows) -> int | None:
  """Index into ROWS of the first row at which the last WINDOW rows no
  longer look like train (by nearest-train distance over the x columns),
  or None if they never do."""
  tbl, _, base = _load(train_csv)
  recent: list[float] = []
  for i, r in enumerate(rows):
    recent.append(_nn(tbl, _full(tbl, r)))
    if len(recent) > WINDOW: recent.pop(0)
    if len(recent) == WINDOW and not ezr.same(base, recent):
      return i
  return None

