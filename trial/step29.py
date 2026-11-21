# step29.py
"""Step 29: spot.  Flag customer records that look wrong (impossible ages,
odd incomes, typos), or that are so unlike anyone else that they should be
checked before a campaign is built on them.

Customers are read as in step21: identifier and text columns are left out,
goals are the numeric columns named with a trailing "+" or "-", and every
other kept column is an input.  A column is numeric if most of its known
cells are numbers.

Uses only the standard library plus ezr.
"""
import math
import random
import ezr

_SAMPLE  = 100    # training customers used to learn how far apart normal rows sit
_SAFE    = 3.0    # robust z-scores up to this are ordinary, not odd
_TYPO    = 5.0    # cost of a non-number cell where a number belongs
_IMPOSS  = 5.0    # cost of a value no real customer can have (e.g. negative age)
_UNSEEN  = 1.0    # cost of a symbol never seen in training
_Q       = 0.9    # a row farther than this training quantile is novel
_FLOOR   = 0.05   # smallest novelty reference, so ratios stay sane
_MAX_AGE = 120    # no real customer is older than this
_DROP    = {"ID", "num", "Rank", "Name", "Nation", "League", "Team", "url"}


def _num(v) -> bool:
  "True if V is a usable finite number (not text, not missing, not a bool)."
  if isinstance(v, bool) or not isinstance(v, (int, float)): return False
  return not isinstance(v, float) or math.isfinite(v)


def _clean(v):
  "A cell as ezr reads it: strings are cast; blanks, '?' and non-finite numbers are unknown."
  if v is None: return "?"
  if isinstance(v, str):
    s = v.strip()
    if s in ("", "?"): return "?"
    v = ezr.atom(s)
  if isinstance(v, float) and not math.isfinite(v): return "?"
  return v


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


def _is_num(rows: list, at: int) -> bool:
  "True if most known cells of column AT are numbers."
  known = [r[at] for r in rows if r[at] != "?"]
  nums  = sum(1 for v in known if _num(v))
  return nums > 0 and 2 * nums >= len(known)


def _prepare(train_csv: str):
  "Load TRAIN_CSV: (header names, kept column positions, cleaned full-width rows), or None if empty."
  raw = list(ezr.csv(train_csv))
  if not raw: return None
  names = [str(h).strip() for h in raw[0]]
  kept  = [at for at, nm in enumerate(names)
           if nm and nm not in _DROP and not nm.endswith("X")]
  rows  = []
  for r in raw[1:]:
    r = [_clean(v) for v in r][:len(names)]
    rows.append(r + ["?"] * (len(names) - len(r)))
  return names, kept, rows


def _bounds(name: str, vs: list[float]):
  "Hard (low, high) limits for a numeric column; None means no limit on that side."
  if name.upper().rstrip("+-!") == "AGE": return (0, _MAX_AGE)
  if vs and min(vs) >= 0:                 return (0, None)
  return (None, None)


def _train(names: list, kept: list, rows: list):
  """What training looks like per kept column.  KIND maps a position to
  'num' or 'sym'; STATS holds (median, robust scale) for numbers, or the
  symbols seen; NUMS holds an ezr.Num per numeric column, for scaling;
  BOUNDS holds the hard limits a numeric value must respect."""
  kind, stats, nums, bounds = {}, {}, {}, {}
  for at in kept:
    if _is_num(rows, at):
      vs = [r[at] for r in rows if _num(r[at])]
      kind[at] = "num"
      nums[at] = ezr.adds(vs, ezr.Num(names[at], at))
      bounds[at] = _bounds(names[at], vs)
      if vs:
        med = _quantile(vs, 0.5)
        iqr = _quantile(vs, 0.75) - _quantile(vs, 0.25)
        stats[at] = (med, iqr / 1.349 if iqr > 0 else _sd(vs))
      else:
        stats[at] = None
    else:
      kind[at]   = "sym"
      stats[at]  = {r[at] for r in rows if r[at] != "?"}
  return kind, stats, nums, bounds


def _full(names: list, kept: list, row) -> list:
  "Lay ROW out full-width in header order, '?' where missing; cells are cleaned."
  q = ["?"] * len(names)
  if isinstance(row, dict):                    # {column name: value}
    stems = {str(k).strip().rstrip("+-!"): v for k, v in row.items()}
    for at, nm in enumerate(names):
      stem = nm.rstrip("+-!")
      if nm in row:        q[at] = row[nm]
      elif stem in stems:  q[at] = stems[stem]
  else:
    seq     = list(row)
    targets = list(range(len(names))) if len(seq) == len(names) else kept
    for at, v in zip(targets, seq): q[at] = v
  return [_clean(v) for v in q]


def _cell(at: int, v, kind: dict, stats: dict, bounds: dict) -> float:
  "How odd one cell V is in column AT, given what training looked like."
  if v == "?": return 0.0
  if kind[at] == "num":
    if not _num(v): return _TYPO                # text where a number belongs
    lo, hi = bounds[at]
    if (lo is not None and v < lo) or (hi is not None and v > hi):
      return _IMPOSS                            # a value no customer can have
    m = stats[at]
    if m is None: return 0.0                    # training had no numbers here
    med, scale = m
    if scale > 0: z = abs(v - med) / scale
    else:         z = 0.0 if v == med else 10.0
    return math.log1p(max(0.0, z - _SAFE))      # graded, so typos stand out
  return 0.0 if v in stats[at] else _UNSEEN     # a symbol never seen before


def _enc(at: int, v, kind: dict, nums: dict):
  "An input cell for distance: None if unknown, symbols as-is, numbers scaled to 0..1."
  if isinstance(v, str) and v == "?": return None
  if kind[at] == "num":
    return ezr.norm(nums[at], v) if _num(v) else None
  return v


def _gap(sym: bool, a, b) -> float:
  "Gap between two encoded cells of one input; unknown is far."
  if a is None or b is None: return 1.0
  if sym: return 0.0 if a == b else 1.0
  return abs(a - b)


def _dist(sym: list[bool], a: list, b: list) -> float:
  "Root mean square gap between two encoded input rows, as ezr.xdist."
  if not sym: return 0.0
  s = sum(_gap(sym[j], a[j], b[j]) ** 2 for j in range(len(sym)))
  return math.sqrt(s / len(sym))


def odd(train_csv: str, rows) -> list[float]:
  """One suspicion score per row of ROWS, in order, goals included.  Higher
  means more likely a glitch or typo, or unlike anything in TRAIN_CSV.  A
  value outside its hard limits (a negative count or spend, or an age past
  _MAX_AGE) adds a large amount.  Each cell far outside the training values
  adds to the score, as does a text cell where a number belongs, and a
  symbol never seen in training.  A row whose inputs are farther from every
  training customer than most training customers are from each other also
  adds its novelty."""
  rows_in = list(rows)
  if not rows_in: return []
  data = _prepare(train_csv)
  if data is None: return [0.0] * len(rows_in)
  names, kept, trows = data
  if not trows: return [0.0] * len(rows_in)

  kind, stats, nums, bounds = _train(names, kept, trows)
  inputs = [at for at in kept if names[at][-1] not in "+-"]
  sym    = [kind[at] == "sym" for at in inputs]
  E      = [[_enc(at, r[at], kind, nums) for at in inputs] for r in trows]
  n      = len(E)

  # What normal looks like for novelty: each sampled training row's nearest other row.
  ref = []
  if inputs and n > 1:
    rng = random.Random(ezr.the.Seed)
    for i in rng.sample(range(n), min(_SAMPLE, n)):
      d = [_dist(sym, E[i], E[j]) for j in range(n) if j != i]
      if d: ref.append(min(d))
  base = max(_quantile(ref, _Q), _FLOOR) if ref else 0.0

  out = []
  for raw in rows_in:
    q = _full(names, kept, raw)
    s = sum(_cell(at, q[at], kind, stats, bounds) for at in kept)
    if inputs and ref:
      e = [_enc(at, q[at], kind, nums) for at in inputs]
      d = min(_dist(sym, e, t) for t in E)
      s += math.log1p(max(0.0, d / base - 1))
    out.append(float(s))
  return out

