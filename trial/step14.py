# step14.py
"""Step 14: flow.  Keep the worthwhile scouting reports as they come in,
and warn when new rows stop looking like the training data.

Uses only the standard library plus ezr.
"""
import bisect
import random
import ezr

_WIN    = 30     # rows per drift block
_MIN    = 10     # smallest trailing block worth testing
_KEEP_Q = 0.25   # keep a row if it is in the best quarter seen so far
_SAMPLE = 200    # train rows used as the flag-novelty reference


def _load(train_csv: str) -> ezr.TBL:
  "Read TRAIN_CSV into an ezr table, keeping file order."
  return ezr.Tbl(ezr.csv(train_csv))


def _full(tbl: ezr.TBL, row) -> ezr.ROW:
  "Lay ROW out full-width, with '?' for anything missing; numeric columns keep only numbers."
  q = ["?"] * len(tbl.names)
  if isinstance(row, dict):                    # {column name: value}
    for c in tbl.cols.values():
      name = tbl.names[c.at]
      if name in row: q[c.at] = row[name]
  elif len(row) == len(tbl.names):             # already full-width
    q = list(row)
  else:                                        # positional, x-column order
    for c, v in zip(tbl.x, row): q[c.at] = v
  out = [ezr.atom(v) if isinstance(v, str) else v for v in q]
  for c in tbl.cols.values():
    v = out[c.at]
    if isinstance(c, ezr.Num) and (isinstance(v, bool)
                                   or not isinstance(v, (int, float))):
      out[c.at] = "?"                          # text where a number belongs
  return out


def _score(tbl: ezr.TBL, q: ezr.ROW) -> float:
  "RMS gap of Q's goals from their ideal: 0 is best, 1 is worst.  Missing is worst."
  gaps = []
  for c in tbl.y:
    v = q[c.at]
    if v == "?" or isinstance(v, str): gaps.append(1.0)
    else: gaps.append(abs(ezr.norm(c, v) - c.goal))
  return ezr.dist(gaps, len(gaps)) if gaps else 0.0


def stream(train_csv: str, rows) -> list[bool]:
  """For each row in ROWS, in order: keep it (True) if its goals are in the
  best quarter seen so far.  The cut is taken from TRAIN_CSV plus the rows
  already seen, and is decided before the current row is learned from."""
  tbl  = _load(train_csv)
  rows = list(rows)
  if not tbl.rows: return [False] * len(rows)
  pool = sorted(_score(tbl, r) for r in tbl.rows)   # what "good" looks like
  keep = []
  for raw in rows:
    s   = _score(tbl, _full(tbl, raw))
    cut = pool[int(_KEEP_Q * (len(pool) - 1))]      # decide before learning
    keep.append(s <= cut)
    bisect.insort(pool, s)                          # then learn from it
  return keep


def drift(train_csv: str, rows) -> int | None:
  """Index into ROWS of the row at which new rows first stop looking like
  TRAIN_CSV, or None.  Rows are tested in blocks of _WIN; a block differs if
  its goal scores or its flag novelty (distance to nearest reference row)
  differ from train's, by ezr.same.  Warn at the end of the second differing
  block in a row, so that single noisy blocks rarely raise a false alarm."""
  tbl  = _load(train_csv)
  rows = list(rows)
  if not tbl.rows or not rows: return None

  ref_g = [_score(tbl, r) for r in tbl.rows]
  rng   = random.Random(ezr.the.Seed)
  idx   = rng.sample(range(len(tbl.rows)), min(_SAMPLE, len(tbl.rows)))
  S     = [tbl.rows[i] for i in idx]
  ref_f = []                                        # train's own novelty
  for i, s in enumerate(S):
    d = [ezr.xdist(tbl, s, t) for j, t in enumerate(S) if j != i]
    if d: ref_f.append(min(d))
  if not ref_f: ref_f = [0.0]

  qs = [_full(tbl, r) for r in rows]
  g  = [_score(tbl, q) for q in qs]
  f  = [min(ezr.xdist(tbl, q, t) for t in S) for q in qs]

  run = 0
  for lo in range(0, len(rows), _WIN):
    hi = min(lo + _WIN, len(rows))
    if hi - lo < _MIN: break
    differs = (not ezr.same(g[lo:hi], ref_g)
               or not ezr.same(f[lo:hi], ref_f))
    run = run + 1 if differs else 0
    if run >= 2: return hi - 1
  return None

