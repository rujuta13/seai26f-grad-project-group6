# step8.py
"""Step 8: fix.  Find the fewest input changes that turn a bad build good,
without changing the frozen inputs.

Uses only the standard library plus ezr.
"""
import math
from itertools import combinations, product
import ezr

_GOOD_Q   = 0.25   # best quarter of training builds counts as good
_MAX_K    = 4      # exhaustive search over up to this many changed inputs
_MAX_VALS = 8      # candidate values per input, evenly spaced by rank
_CAP      = 20000  # most candidate builds to score before falling back


def _load(train_csv: str) -> ezr.TBL:
  "Read TRAIN_CSV into an ezr table, keeping file order."
  return ezr.Tbl(ezr.csv(train_csv))


def _quantile(xs: list[float], p: float) -> float:
  "The p-th quantile of XS, linearly interpolated.  XS must be non-empty."
  xs     = sorted(xs)
  i      = p * (len(xs) - 1)
  lo, hi = math.floor(i), math.ceil(i)
  return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def _sortkey(v) -> tuple:
  "Sort numbers before strings, so mixed cells never compare across types."
  return (isinstance(v, str), v)


def _full(tbl: ezr.TBL, row) -> ezr.ROW:
  "Lay ROW out full-width, with '?' for anything missing; strings are cast."
  q = ["?"] * len(tbl.names)
  if isinstance(row, dict):                    # {column name: value}
    for c in tbl.cols.values():
      name = tbl.names[c.at]
      if name in row: q[c.at] = row[name]
  elif len(row) == len(tbl.names):             # already full-width
    q = list(row)
  else:                                        # positional, x-column order
    for c, v in zip(tbl.x, row): q[c.at] = v
  return [ezr.atom(v) if isinstance(v, str) else v for v in q]


def _inputs(tbl: ezr.TBL, q: ezr.ROW) -> dict:
  "The input cells of full-width row Q, keyed by column name."
  return {c.txt: q[c.at] for c in tbl.x}


def _enc(c: ezr.COL, v) -> float | str | None:
  "A cell for distance: None if unknown, symbols as-is, numbers scaled to 0..1."
  if v == "?": return None
  return ezr.norm(c, v) if isinstance(c, ezr.Num) else v


def fix(train_csv: str, row, frozen=()) -> dict:
  """Return the inputs of ROW with as few values changed as possible, so that
  its predicted goals fall in the best quarter of TRAIN_CSV.  Columns named in
  FROZEN never change.  ROW is a bad build with its goals removed.  The
  predicted goals of a candidate are the median goals of its nearest training
  rows, as in step2.  Changes are searched exhaustively by count (up to
  _MAX_K); if none succeeds, a greedy search is used and its best effort is
  returned."""
  tbl  = _load(train_csv)
  rows = tbl.rows
  if isinstance(frozen, str): frozen = (frozen,)
  frozen = set(frozen)
  base = _full(tbl, row)
  if not rows or not tbl.y or not tbl.x:
    return _inputs(tbl, base)

  xcols = tbl.x
  sym   = [not isinstance(c, ezr.Num) for c in xcols]
  E     = [[_enc(c, r[c.at]) for c in xcols] for r in rows]
  cut   = _quantile([ezr.ydist(tbl, r) for r in rows], _GOOD_Q)
  k     = min(len(rows), max(5, int(math.sqrt(len(rows)))))

  def dist_enc(eq: list, e: list) -> float:
    "Root mean square gap between two encoded input rows, as ezr.xdist."
    s = 0.0
    for j, (a, b) in enumerate(zip(eq, e)):
      if a is None or b is None: g = 1
      elif sym[j]:               g = int(a != b)
      else:                      g = abs(a - b)
      s += g * g
    return math.sqrt(s / len(xcols))

  def predict(q: ezr.ROW) -> float:
    "Predicted distance from the ideal for full-width row Q; lower is better."
    eq    = [_enc(c, q[c.at]) for c in xcols]
    order = sorted(range(len(rows)),
                   key=lambda i: dist_enc(eq, E[i]))[:k]
    gaps  = []
    for c in tbl.y:
      vs = [rows[i][c.at] for i in order if rows[i][c.at] != "?"]
      if not vs:
        vs = [r[c.at] for r in rows if r[c.at] != "?"]
      if vs: gaps.append(abs(ezr.norm(c, _quantile(vs, 0.5)) - c.goal))
      else:  gaps.append(1.0)
    return ezr.dist(gaps, len(gaps))

  movable = [c for c in xcols if c.txt not in frozen]
  vals: dict[int, list] = {}                   # candidate values per input
  for c in xcols:
    seen = sorted({r[c.at] for r in rows if r[c.at] != "?"}, key=_sortkey)
    if len(seen) > _MAX_VALS:
      seen = [seen[round(i * (len(seen) - 1) / (_MAX_VALS - 1))]
              for i in range(_MAX_VALS)]
    vals[c.at] = seen
  opts = {c.at: [v for v in vals[c.at] if v != base[c.at]] for c in movable}

  def candidates():
    "Yield (number of changed inputs, full-width row), fewest changes first."
    for size in range(0, min(len(movable), _MAX_K) + 1):
      for cols in combinations(movable, size):
        for new in product(*(opts[c.at] for c in cols)):
          q = list(base)
          for c, v in zip(cols, new): q[c.at] = v
          yield size, q

  # Exhaustive by count: the first size with any good build wins.
  found = None                                 # (size, predicted, row)
  evals = 0
  for size, q in candidates():
    if found is not None and size != found[0]: break
    y = predict(q)
    evals += 1
    if y <= cut and (found is None or y < found[1]):
      found = (size, y, q)
    if evals >= _CAP: break
  if found is not None:
    return _inputs(tbl, found[2])

  # Greedy fallback: keep taking the single change that helps most.
  q   = list(base)
  cur = predict(q)
  while cur > cut:
    step = None
    for c in movable:
      for v in vals[c.at]:
        if v == q[c.at]: continue
        trial = list(q)
        trial[c.at] = v
        y = predict(trial)
        if step is None or y < step[0]: step = (y, trial)
    if step is None or step[0] >= cur: break
    cur, q = step
  return _inputs(tbl, q)

