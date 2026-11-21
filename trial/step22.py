# step22.py
"""Step 22: guess.  Before a campaign, guess each goal of a new customer from
the customer's other numbers, with a range calibrated on the training data:
how many purchases they will make online and in store, and whether they will
answer the next campaign (RESPONSE+).

The estimate is the median goal value of the training rows nearest to the
customer.  The range is that estimate widened by the 90th percentile of how
far the same kind of guess missed on training rows held out of their own
neighbourhood, and clamped to the goal values seen in training.

Uses only the standard library plus ezr.
"""
import math
import random
import ezr

_CAL   = 150     # training rows used to calibrate the range
_COVER = 0.9     # the range should hold about this share of the time


def guess(train_csv: str, row) -> dict[str, tuple[float, float, float]]:
  """For each goal: (estimate, low, high).  ROW has its goal columns removed.
  The estimate is the median goal value of the training rows nearest to ROW.
  LOW and HIGH bracket it by the calibrated miss, clamped to the goal values
  seen in training, so the range is never wider than the data allows."""

  def num(v) -> bool:
    "True if V is a usable finite number (not text, not missing, not a bool)."
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))

  def clean(v):
    "A cell as ezr reads it: strings are cast; blanks, '?' and non-finite numbers are unknown."
    if v is None: return "?"
    if isinstance(v, str):
      s = v.strip()
      if s in ("", "?"): return "?"
      v = ezr.atom(s)
      if isinstance(v, str): return v
    if isinstance(v, float) and not math.isfinite(v): return "?"
    return v

  def quantile(xs, p: float) -> float:
    "The p-th quantile of XS, linearly interpolated.  XS must be non-empty."
    xs     = sorted(xs)
    i      = p * (len(xs) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)

  def encode(c, v):
    "An input cell for distance: None if unknown, symbols as-is, numbers scaled to 0..1."
    if v == "?": return None
    if isinstance(c, ezr.Num): return ezr.norm(c, v) if num(v) else None
    return v

  def gap(c, a, b) -> float:
    "Gap between two encoded cells of column C; unknown is far."
    if a is None or b is None: return 1.0
    if isinstance(c, ezr.Num): return abs(a - b)
    return 0.0 if a == b else 1.0

  # Training table: every cell cleaned, every row padded to full width.
  raw = list(ezr.csv(train_csv))
  if not raw: return {}
  names = [str(h).strip() for h in raw[0]]
  body  = []
  for r in raw[1:]:
    r = [clean(v) for v in r][:len(names)]
    body.append(r + ["?"] * (len(names) - len(r)))
  tbl   = ezr.Tbl([names] + body)
  rows  = tbl.rows
  xs    = tbl.x
  goals = list(tbl.y)
  if not goals: return {}
  if not rows:  return {c.txt: (0.0, 0.0, 0.0) for c in goals}

  n = len(rows)
  k = min(n, max(5, int(math.sqrt(n))))
  E = [[encode(c, r[c.at]) for c in xs] for r in rows]

  # The new customer's input cells, laid out in header order.
  q = ["?"] * len(names)
  if isinstance(row, dict):                    # {column name: value}
    stems = {str(key).strip().rstrip("+-!"): v for key, v in row.items()}
    for c in xs:
      stem = names[c.at].rstrip("+-!")
      if stem in stems: q[c.at] = stems[stem]
  else:
    seq = list(row)
    if len(seq) == len(names):                 # already full-width
      q = seq
    else:                                      # positional, x-column order
      for c, v in zip(xs, seq): q[c.at] = v
  eq = [encode(c, clean(q[c.at])) for c in xs]

  def dist(a, b) -> float:
    "Root mean square gap between two encoded input rows, as ezr.xdist."
    if not xs: return 0.0
    return math.sqrt(sum(gap(c, x, y) ** 2 for c, x, y in zip(xs, a, b))
                     / len(xs))

  def near(e, skip: int | None = None) -> list[int]:
    "Indices of the K training rows nearest encoded row E, leaving out SKIP."
    order = sorted((i for i in range(n) if i != skip),
                   key=lambda i: (dist(e, E[i]), i))
    return order[:k]

  rng   = random.Random(ezr.the.Seed)          # local rng: no global side effects
  cal   = rng.sample(range(n), min(_CAL, n))
  cnear = {i: near(E[i], skip=i) for i in cal} # held out of their own neighbourhood
  qn    = near(eq)

  out = {}
  for c in goals:
    gv = {}                                    # row index -> known goal value
    for i, r in enumerate(rows):
      v = r[c.at]
      if num(v): gv[i] = float(v)
    if not gv:
      out[c.txt] = (0.0, 0.0, 0.0)
      continue
    allv = list(gv.values())

    def estimate(idxs: list[int]) -> float:
      "Median goal value of the known goals among IDXS (all rows if none)."
      vs = [gv[i] for i in idxs if i in gv] or allv
      return quantile(vs, 0.5)

    est   = estimate(qn)
    resid = [abs(gv[i] - estimate(cnear[i])) for i in cal if i in gv]
    half  = (quantile(resid, _COVER) if resid
             else (max(allv) - min(allv)) / 2)
    lo    = max(min(allv), est - half)
    hi    = min(max(allv), est + half)
    out[c.txt] = (float(est), float(min(lo, est)), float(max(hi, est)))
  return out

