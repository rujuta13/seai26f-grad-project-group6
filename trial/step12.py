# step12.py
"Step 12: guess. Estimate each goal of a build from its nearest training builds, with a range we are fairly sure of."
import math
import os
import ezr

K    = 20     # nearest training builds consulted
MINK = 5      # fewer known neighbours than this: widen to all training builds
LO   = 0.10   # low end of the "fairly sure" range
HI   = 0.90   # high end of the "fairly sure" range

_CACHE: dict = {}   # (path, mtime) -> (table, input columns used for distance)

NAN3 = (math.nan, math.nan, math.nan)

def _key(train_csv: str) -> tuple:
  "Identity of TRAIN_CSV: its absolute path and when it last changed."
  path = os.path.abspath(os.path.expanduser(train_csv))
  return (path, os.path.getmtime(path))

def _inputs(tbl) -> list:
  """Input columns used for distance: all numbers, and symbols that repeat.
  A symbol where every value is distinct is an identifier (a row number or
  url), so it says nothing about how builds relate."""
  return [c for c in tbl.x
          if type(c) is ezr.Num or len(c.has) < c.n]

def _load(train_csv: str):
  "Train table and its distance columns, cached per file version."
  key = _key(train_csv)
  if key not in _CACHE:
    tbl = ezr.Tbl(ezr.csv(train_csv))
    _CACHE[key] = (tbl, _inputs(tbl))
  return _CACHE[key]

def _coerce(v):
  "Strings become atoms; other values pass through."
  return ezr.atom(v) if isinstance(v, str) else v

def _isnum(v) -> bool:
  "True for a usable finite number."
  return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def _full(tbl, row) -> list:
  """Lift ROW into a full-width row. Accepts a {column name: value} dict,
  a full-width list, or a list with goal columns removed (header order).
  Any cell not given is '?'. Non-numeric cells in numeric columns become '?'."""
  names = tbl.names
  out   = ["?"] * len(names)
  if isinstance(row, dict):
    for at, n in enumerate(names):
      if n in row: out[at] = _coerce(row[n])
  else:
    row = list(row)
    if len(row) == len(names):
      out = [_coerce(v) for v in row]
    else:
      keep = [i for i, n in enumerate(names) if n[-1] not in "+-!"]
      for i, v in zip(keep, row): out[i] = _coerce(v)
  for c in tbl.x:
    if type(c) is ezr.Num and not _isnum(out[c.at]):
      out[c.at] = "?"
  return out

def _pct(xs: list, p: float) -> float:
  "Percentile P (0..1) of sorted XS, by linear interpolation."
  i  = p * (len(xs) - 1)
  lo = int(i)
  hi = min(lo + 1, len(xs) - 1)
  return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)

def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
  """Return {goal: (estimate, low, high)} for every goal, goal names as in the
  header (with their suffix). ROW has its goal columns removed. The estimate
  is the median goal value of the K training builds nearest to ROW over the
  input columns; low..high is the LO..HI percentile band of those neighbours.
  A goal no nearby build has gets a band from all training builds. A goal
  with no data at all, or a non-numeric goal, gets NaN."""
  tbl, cols = _load(train_csv)
  q         = _full(tbl, row)
  rows      = tbl.rows

  def dist2(r) -> float:
    return sum(ezr.gap(c, q[c.at], r[c.at]) ** 2 for c in cols)

  near = sorted(rows, key=dist2)[:K]
  out  = {}
  for c in tbl.y:
    if type(c) is not ezr.Num:             # symbolic goal: no number to give
      out[c.txt] = NAN3
      continue
    xs = sorted(r[c.at] for r in near if _isnum(r[c.at]))
    if len(xs) < MINK:                      # too few neighbours know this goal
      xs = sorted(r[c.at] for r in rows if _isnum(r[c.at]))
    if not xs:
      out[c.txt] = NAN3
      continue
    out[c.txt] = (float(_pct(xs, 0.5)),
                  float(_pct(xs, LO)),
                  float(_pct(xs, HI)))
  return out

