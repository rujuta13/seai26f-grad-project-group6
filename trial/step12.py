# step12.py
"""
step12: guess, with confidence.  For each goal of a row with its goals
removed, return (estimate, low, high).  The estimate is the median goal
value among training rows in the same family.  low..high is the middle 90%
of those values, our "fairly sure" range.  When the family is thin, the
range comes from every training row instead, so it is wider and says so.
Reuses step2 (model, _full, _q) and ezr (leaf).
"""
import math
import ezr
import step2

FAIR    = (0.05, 0.95)   # "fairly sure": the middle 90% of the values seen
MIN_VAL = 5               # fewer numeric family values than this: use every row
THIN    = 30              # fewer family rows than this: range from every row


def _nums(rows, at) -> list[float]:
  "Sorted finite numbers in column AT of ROWS; '?' and text are skipped."
  return sorted(float(r[at]) for r in rows
                if isinstance(r[at], (int, float)) and math.isfinite(r[at]))


def _interval(fam, rows, at) -> tuple[float, float, float]:
  "(estimate, low, high) for the goal in column AT, from FAM, else ROWS."
  fv   = _nums(fam, at)
  allv = _nums(rows, at)
  if not allv:
    return (float("nan"),) * 3
  est_src = fv if len(fv) >= MIN_VAL else allv     # estimate: family if it can be trusted
  est     = step2._q(est_src, 0.5)
  thin    = len(fam) < THIN or len(fv) < MIN_VAL
  rng_src = allv if thin else fv                    # thin family: judge by the whole table
  lo, hi  = step2._q(rng_src, FAIR[0]), step2._q(rng_src, FAIR[1])
  return (est, min(lo, est), max(hi, est))          # the estimate always sits inside


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
  """For every goal: (estimate, low, high). Keys are the goal column names,
  as in the csv header. ROW has its goal columns removed."""
  tbl, node = step2._model(train_csv)
  full = step2._full(tbl, row)                      # goal cells are '?'
  fam  = ezr.leaf(node, full).rows                  # training rows in the same family
  return {c.txt: _interval(fam, tbl.rows, c.at) for c in tbl.y}

