# step2.py
"""Guess a flag set's goals (speed and CPU cost), with a range of doubt.

The training rows nearest the query, in x-space, vote. The estimate is
their mean. The range is their 10th to 90th percentile, so a wide range
means the neighbours disagree and the guess is shaky.
"""
import ezr

_cache: dict = {}   # train_csv -> parsed table, so repeat calls stay cheap


def _load(train_csv: str):
  "Parse TRAIN_CSV once, then reuse it."
  if train_csv not in _cache:
    _cache[train_csv] = ezr.Tbl(ezr.csv(train_csv))
  return _cache[train_csv]


def _cell(v):
  "Strings from a csv or a dict become atoms; other values pass through."
  return ezr.atom(v) if isinstance(v, str) else v


def _full(tbl, row) -> list:
  """Lay ROW out as a full-width ezr row, aligned with tbl.names.
  ROW may be a dict {column name: value}, or a list of x cells in x
  order (goal columns already removed). Missing cells become "?"."""
  full = ["?"] * len(tbl.names)
  if isinstance(row, dict):
    for at, name in enumerate(tbl.names):
      if name in row: full[at] = _cell(row[name])
  elif len(row) == len(tbl.names):
    full = [_cell(v) for v in row]
  else:
    for c, v in zip(tbl.x, row): full[c.at] = _cell(v)
  return full


def _pct(xs: list, p: float) -> float:
  "The p-th percentile of XS (0..1), by linear interpolation."
  xs = sorted(xs)
  i  = p * (len(xs) - 1)
  lo = int(i)
  hi = min(lo + 1, len(xs) - 1)
  return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def _known(v) -> bool:
  "True for a real number (not '?')."
  return isinstance(v, (int, float))


def guess(train_csv: str, row) -> dict[str, tuple[float, float, float]]:
  """Return {goal: (estimate, low, high)} for every goal.
  Goal keys are the full header names, e.g. 'Performance-'.
  The nearest training rows (by xdist over the x columns) vote:
  estimate = their mean, low..high = their 10th..90th percentile."""
  tbl  = _load(train_csv)
  rows = tbl.rows
  q    = _full(tbl, row)
  k    = max(5, int(len(rows) ** .5))   # neighbourhood size grows with data
  near = sorted(rows, key=lambda r: ezr.xdist(tbl, r, q))[:k]
  out  = {}
  for g in tbl.y:
    if g.txt[-1] not in "+-": continue  # only goals; skip the klass column
    vals = [r[g.at] for r in near if _known(r[g.at])]
    if not vals:                        # no known value nearby: use all rows
      vals = [r[g.at] for r in rows if _known(r[g.at])]
    if not vals: raise ValueError(f"no known values for {g.txt}")
    est = sum(vals) / len(vals)
    lo, hi = _pct(vals, .1), _pct(vals, .9)
    out[g.txt] = (est, min(lo, est), max(hi, est))
  return out

