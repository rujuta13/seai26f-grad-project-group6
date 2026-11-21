# step2.py
"Step 2: guess each goal of a flag set from its nearest training rows."
import ezr

K  = 20     # nearest training rows to consult
LO = 0.10   # low end of the "fairly sure" range
HI = 0.90   # high end of the "fairly sure" range

_CACHE: dict = {}

def _load(train_csv: str):
  "Table of TRAIN_CSV, cached so repeated guesses are cheap."
  if train_csv not in _CACHE:
    _CACHE[train_csv] = ezr.Tbl(ezr.csv(train_csv))
  return _CACHE[train_csv]

def _pct(xs: list, p: float) -> float:
  "Percentile P (0..1) of sorted XS, by linear interpolation."
  i  = p * (len(xs) - 1)
  lo = int(i)
  hi = min(lo + 1, len(xs) - 1)
  return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)

def _coerce(v):
  "Strings become atoms; other values pass through."
  return ezr.atom(v) if isinstance(v, str) else v

def _full(tbl, row) -> list:
  """Lift ROW (goal columns removed) into a full-width row.
  Goal cells, and any cells not given, are '?'."""
  names = tbl.names
  out   = ["?"] * len(names)
  if isinstance(row, dict):                  # {column name: value}
    for c in tbl.x:
      if c.txt in row: out[c.at] = _coerce(row[c.txt])
  else:                                      # list, in header order, goals removed
    keep = [i for i, n in enumerate(names) if n[-1] not in "+-!"]
    for i, v in zip(keep, row): out[i] = _coerce(v)
  return out

def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
  """Return {goal: (estimate, low, high)} for every goal.
  The estimate is the median goal value of the K training rows
  nearest to ROW over the input columns; low..high is the
  LO..HI percentile band of those same neighbours."""
  tbl  = _load(train_csv)
  q    = _full(tbl, row)
  rows = tbl.rows
  near = sorted(rows, key=lambda r: ezr.xdist(tbl, q, r))[:K]
  out  = {}
  for c in tbl.y:
    xs = sorted(r[c.at] for r in near if r[c.at] != "?")
    if not xs:                               # no neighbour has this goal
      xs = sorted(r[c.at] for r in rows if r[c.at] != "?")
    if not xs:
      out[c.txt] = (0.0, 0.0, 0.0)
      continue
    lo, est, hi = _pct(xs, LO), _pct(xs, 0.5), _pct(xs, HI)
    out[c.txt] = (float(est), float(lo), float(hi))
  return out

