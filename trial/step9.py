# step9.py
"""Step 9: spot.  Flag benchmark rows that look like glitches or typos, or
that are unlike anything in the training data.

Uses only the standard library plus ezr.
"""
import math
import random
import ezr

_SAMPLE = 100     # train rows used to learn how far apart normal rows sit
_SAFE   = 3.0     # robust z-scores up to this are ordinary, not odd
_TYPO   = 5.0     # cost of a text cell where a number belongs
_UNSEEN = 1.0     # cost of a symbol never seen in training
_Q      = 0.9     # a row farther than this train quantile is novel


def _load(train_csv: str) -> ezr.TBL:
  "Read TRAIN_CSV into an ezr table, keeping file order."
  return ezr.Tbl(ezr.csv(train_csv))


def _quantile(xs: list[float], p: float) -> float:
  "The p-th quantile of XS, linearly interpolated.  XS must be non-empty."
  xs     = sorted(xs)
  i      = p * (len(xs) - 1)
  lo, hi = math.floor(i), math.ceil(i)
  return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def _sd(xs: list[float]) -> float:
  "Sample standard deviation of XS, or 0 if fewer than two values."
  n = len(xs)
  if n < 2: return 0.0
  mu = sum(xs) / n
  return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def _full(tbl: ezr.TBL, row) -> ezr.ROW:
  "Lay ROW out full-width (one cell per column of TBL), '?' where missing; strings are cast."
  q = ["?"] * len(tbl.names)
  if isinstance(row, dict):                    # {column name: value}
    stripped = {str(k).rstrip("+-!"): v for k, v in row.items()}
    for c in tbl.cols.values():
      name = tbl.names[c.at]
      if name in row: q[c.at] = row[name]
      elif name.rstrip("+-!") in stripped:
        q[c.at] = stripped[name.rstrip("+-!")]
  elif len(row) == len(tbl.names):             # already full-width
    q = list(row)
  elif len(row) == len(tbl.cols):              # every non-X column, file order
    cols = sorted(tbl.cols.values(), key=lambda c: c.at)
    for c, v in zip(cols, row): q[c.at] = v
  else:                                        # positional, x-column order
    for c, v in zip(tbl.x, row): q[c.at] = v
  return [ezr.atom(v) if isinstance(v, str) else v for v in q]


def _model(tbl: ezr.TBL) -> dict:
  "Per column, what training looks like: a robust (centre, scale) for numbers, or the symbols seen."
  out = {}
  for c in tbl.cols.values():
    if isinstance(c, ezr.Num):
      vs = [r[c.at] for r in tbl.rows if isinstance(r[c.at], (int, float))]
      if not vs:
        out[c.at] = None
      else:
        med = _quantile(vs, 0.5)
        iqr = _quantile(vs, 0.75) - _quantile(vs, 0.25)
        out[c.at] = (med, iqr / 1.349 if iqr > 0 else _sd(vs))
    else:
      out[c.at] = {r[c.at] for r in tbl.rows if r[c.at] != "?"}
  return out


def _cell(c: ezr.COL, v, m) -> float:
  "How odd one cell V is in column C, given what training looked like (M)."
  if v == "?": return 0.0
  if isinstance(c, ezr.Num):
    if isinstance(v, str): return _TYPO        # text where a number belongs
    if m is None: return 0.0                   # training had no numbers here
    med, scale = m
    if scale > 0: z = abs(v - med) / scale
    else:         z = 0.0 if v == med else 10.0
    return math.log1p(max(0.0, z - _SAFE))     # graded, so typos stand out
  return 0.0 if v in m else _UNSEEN            # a symbol never seen before


def odd(train_csv: str, rows) -> list[float]:
  """One suspicion score per row of ROWS, in order, goals included.  Higher
  means more likely a glitch or typo, or unlike anything in TRAIN_CSV.  Each
  cell far outside the training values adds to the score, as does a text
  cell where a number belongs, and a symbol never seen in training.  A row
  whose flags are farther from every training row than most training rows
  are from each other also adds its novelty."""
  tbl  = _load(train_csv)
  rows = list(rows)
  if not rows: return []
  if not tbl.rows: return [0.0] * len(rows)

  m    = _model(tbl)
  cols = sorted(tbl.cols.values(), key=lambda c: c.at)

  def clean(q: ezr.ROW) -> ezr.ROW:
    "Q with text in numeric input columns blanked, so that distances can be taken."
    q = list(q)
    for c in tbl.x:
      if isinstance(c, ezr.Num) and isinstance(q[c.at], str):
        q[c.at] = "?"
    return q

  def nearest(q: ezr.ROW) -> float:
    "Distance from full-width Q to its nearest training row, over the x columns."
    return min(ezr.xdist(tbl, q, t) for t in tbl.rows)

  # What normal looks like for novelty: each sampled train row's nearest other train row.
  ref = []
  if tbl.x:
    rng = random.Random(ezr.the.Seed)
    idx = rng.sample(range(len(tbl.rows)), min(_SAMPLE, len(tbl.rows)))
    for i in idx:
      d = [ezr.xdist(tbl, tbl.rows[i], t)
           for j, t in enumerate(tbl.rows) if j != i]
      if d: ref.append(min(d))
  ref_q = _quantile(ref, _Q) if ref else 0.0

  out = []
  for raw in rows:
    q = _full(tbl, raw)
    s = sum(_cell(c, q[c.at], m[c.at]) for c in cols)
    if tbl.x and ref:
      d  = nearest(clean(q))
      s += math.log1p(max(0.0, d / (ref_q + 1e-32) - 1))
    out.append(s)
  return out

