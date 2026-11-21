# step19.py
"""Step 19: spot.  Flag player records that look like data-entry mistakes,
or that are so unusual their ratings should be looked at twice.

Players are read as in step16: identifier and text columns (num, Rank,
Name, Nation, League, Team, url) are left out, and every other kept column
is scored, goals included.  A numeric column keeps its numbers, and a text
cell where a number belongs is a typo signal.

Uses only the standard library plus ezr.
"""
import math
import random
import ezr

_SAMPLE = 100     # train rows used to learn how far apart normal rows sit
_SAFE   = 3.0     # robust z-scores up to this are ordinary, not odd
_TYPO   = 5.0     # cost of a non-number cell where a number belongs
_UNSEEN = 1.0    # cost of a symbol never seen in training
_Q      = 0.9     # a row farther than this train quantile is novel
_FLOOR  = 0.05    # smallest novelty reference, so ratios stay sane
_DROP   = {"num", "Rank", "Name", "Nation", "League", "Team", "url"}


def _num(v) -> bool:
  "True if V is a usable number (not text, not missing, not a bool)."
  return isinstance(v, (int, float)) and not isinstance(v, bool)


def _clean(v):
  "A cell as ezr reads it: strings are cast, blanks and None become '?'."
  if v is None: return "?"
  if isinstance(v, str):
    s = v.strip()
    return ezr.atom(s) if s else "?"
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
  "Load TRAIN_CSV: (header names, kept column positions, cleaned rows)."
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


def _train(names: list, kept: list, rows: list):
  """What training looks like per kept column.  KIND maps a position to
  'num' or 'sym'; STATS holds (median, robust scale) for numbers, or the
  symbols seen; NUMS holds an ezr.Num per numeric column, for scaling."""
  kind, stats, nums = {}, {}, {}
  for at in kept:
    if _is_num(rows, at):
      vs = [r[at] for r in rows if _num(r[at])]
      kind[at] = "num"
      nums[at] = ezr.adds(vs, ezr.Num(names[at], at))
      if vs:
        med = _quantile(vs, 0.5)
        iqr = _quantile(vs, 0.75) - _quantile(vs, 0.25)
        stats[at] = (med, iqr / 1.349 if iqr > 0 else _sd(vs))
      else:
        stats[at] = None
    else:
      kind[at]  = "sym"
      stats[at] = {r[at] for r in rows if r[at] != "?"}
  return kind, stats, nums


def _full(names: list, kept: list, row) -> list:
  "Lay ROW out full-width in header order, '?' where missing; cells are cleaned."
  q = ["?"] * len(names)
  if isinstance(row, dict):                    # {column name: value}
    by_stem = {str(k).strip().rstrip("+-!"): v for k, v in row.items()}
    for at, name in enumerate(names):
      stem = name.rstrip("+-!")
      if name in row:          q[at] = row[name]
      elif stem in by_stem:    q[at] = by_stem[stem]
  else:
    seq     = list(row)
    targets = list(range(len(names))) if len(seq) == len(names) else kept
    for at, v in zip(targets, seq): q[at] = v
  return [_clean(v) for v in q]


def _cell(at: int, v, kind: dict, stats: dict) -> float:
  "How odd one cell V is in column AT, given what training looked like."
  if v == "?": return 0.0
  if kind[at] == "num":
    if not _num(v): return _TYPO               # text where a number belongs
    m = stats[at]
    if m is None: return 0.0                   # training had no numbers here
    med, scale = m
    if scale > 0: z = abs(v - med) / scale
    else:         z = 0.0 if v == med else 10.0
    return math.log1p(max(0.0, z - _SAFE))     # graded, so typos stand out
  return 0.0 if v in stats[at] else _UNSEEN    # a symbol never seen before


def _enc(at: int, v, kind: dict, nums: dict):
  "An input cell for distance: None if unknown, symbols as-is, numbers scaled to 0..1."
  if v == "?": return None
  if kind[at] == "num":
    return ezr.norm(nums[at], v) if _num(v) else None
  return v


def _gap(sym: bool, a, b) -> float:
  "Gap between two encoded cells of one column; unknown is far."
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
  means more likely a glitch or typo, or unlike anything in TRAIN_CSV.  Each
  cell far outside the training values adds to the score, as does a text
  cell where a number belongs, and a symbol never seen in training.  A row
  whose inputs are farther from every training row than most training rows
  are from each other also adds its novelty."""
  rows_in = list(rows)
  if not rows_in: return []
  data = _prepare(train_csv)
  if data is None: return [0.0] * len(rows_in)
  names, kept, trows = data
  if not trows: return [0.0] * len(rows_in)

  kind, stats, nums = _train(names, kept, trows)
  inputs = [at for at in kept if names[at][-1] not in "+-"]
  sym    = [kind[at] == "sym" for at in inputs]
  E      = [[_enc(at, r[at], kind, nums) for at in inputs] for r in trows]
  n      = len(E)

  # What normal looks like for novelty: each sampled train row's nearest other row.
  ref = []
  if inputs and n > 1:
    rng = random.Random(ezr.the.Seed)
    idx = rng.sample(range(n), min(_SAMPLE, n))
    for i in idx:
      d = [_dist(sym, E[i], E[j]) for j in range(n) if j != i]
      if d: ref.append(min(d))
  base = max(_quantile(ref, _Q), _FLOOR) if ref else 0.0

  out = []
  for raw in rows_in:
    q = _full(names, kept, raw)
    s = 0.0
    for at in kept:
      s += _cell(at, q[at], kind, stats)
    if inputs and ref:
      e = [_enc(at, q[at], kind, nums) for at in inputs]
      d = min(_dist(sym, e, t) for t in E)
      s += math.log1p(max(0.0, d / base - 1))
    out.append(float(s))
  return out

