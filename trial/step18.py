# step18.py
"""Step 18: fix.  Find the fewest input changes that turn a bad player into a
good one, without changing the frozen inputs.

Players are read as in step16: identifier and text columns are left out,
goals are the numeric columns named with a trailing "+" or "-", and every
other kept column is an input.  A player is good when its goals, guessed
from the nearest training players as in step2, fall in the best quarter of
the training players.

Uses only the standard library plus ezr.
"""
import math
from itertools import combinations, product
import ezr

_GOOD_Q    = 0.25   # best quarter of training players counts as good
_MAX_VALS  = 5      # candidate values per input, spread evenly by rank
_PAIR_COLS = 12     # two-change search only over the columns that help most
_STEPS     = 20     # most greedy passes when no small change set works
_DROP      = {"num", "Rank", "Name", "Nation", "League", "Team", "url"}


def _num(v) -> bool:
  "True if V is a usable number (not text, not missing, not a bool)."
  return isinstance(v, (int, float)) and not isinstance(v, bool)


def _clean(v):
  "A cell as ezr reads it: strings are cast, blanks and None become '?'."
  if v is None: return "?"
  if isinstance(v, str): return ezr.atom(v) if v.strip() else "?"
  return v


def _sortkey(v) -> tuple:
  "Sort numbers before strings, so mixed cells never compare across types."
  return (isinstance(v, str), v)


def _quantile(xs: list[float], p: float) -> float:
  "The p-th quantile of XS, linearly interpolated.  XS must be non-empty."
  xs     = sorted(xs)
  i      = p * (len(xs) - 1)
  lo, hi = math.floor(i), math.ceil(i)
  return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def _is_num(rows: list, at: int) -> bool:
  "True if most known cells of column AT are numbers."
  known = [r[at] for r in rows if r[at] != "?"]
  nums  = sum(1 for v in known if _num(v))
  return nums > 0 and 2 * nums >= len(known)


def _model(names: list, rows: list, at: int) -> ezr.Num:
  "An ezr.Num summarising the numeric cells of column AT; its goal sign comes from its name."
  return ezr.adds([r[at] for r in rows if _num(r[at])],
                  ezr.Num(names[at], at))


def _prepare(train_csv: str):
  """Load TRAIN_CSV: (names, rows, goals, inputs, kept).  GOALS is a list of
  (position, model) for numeric goals; INPUTS is a list of (position, model),
  where the model is None for a symbolic input; KEPT lists every usable
  column position, goals included."""
  raw = list(ezr.csv(train_csv))
  if not raw: return None
  names = [str(h).strip() for h in raw[0]]
  rows  = []
  for r in raw[1:]:
    r = [_clean(v) for v in r][:len(names)]
    rows.append(r + ["?"] * (len(names) - len(r)))
  kept = [at for at, nm in enumerate(names)
          if nm and nm not in _DROP and not nm.endswith("X")]
  goals, inputs = [], []
  for at in kept:
    nm      = names[at]
    numeric = _is_num(rows, at)
    if nm[-1] in "+-":
      if numeric: goals.append((at, _model(names, rows, at)))
    else:
      inputs.append((at, _model(names, rows, at) if numeric else None))
  return names, rows, goals, inputs, kept


def _full(names: list, kept: list, inputs: list, row) -> list:
  "Lay ROW out full-width in header order, '?' where missing; cells are cast."
  q = ["?"] * len(names)
  if isinstance(row, dict):                    # {column name: value}
    by_stem = {str(k).strip().rstrip("+-!"): v for k, v in row.items()}
    for at, name in enumerate(names):
      stem = name.rstrip("+-!")
      if name in row:        q[at] = row[name]
      elif stem in by_stem:  q[at] = by_stem[stem]
  else:
    seq = list(row)
    if len(seq) == len(names):      targets = list(range(len(names)))
    elif len(seq) == len(inputs):   targets = [at for at, _ in inputs]
    else:                           targets = kept
    for at, v in zip(targets, seq): q[at] = v
  return [_clean(v) for v in q]


def _levels(rows: list, at: int, numeric: bool) -> list:
  "Candidate values for column AT: its distinct known values, spread evenly by rank."
  seen = {r[at] for r in rows
          if r[at] != "?" and (not numeric or _num(r[at]))}
  out  = sorted(seen, key=_sortkey)
  if len(out) > _MAX_VALS:
    out = [out[round(t * (len(out) - 1) / (_MAX_VALS - 1))]
           for t in range(_MAX_VALS)]
  return out


def _enc(m, v):
  "A cell for distance: None if unknown, symbols as-is, numbers scaled to 0..1."
  if isinstance(v, str) and v == "?": return None
  if m is None: return v
  return ezr.norm(m, v) if _num(v) else None


def _g2(sym: bool, a, b) -> float:
  "Squared gap between two encoded cells, as ezr.xdist; unknown is far."
  if a is None or b is None: g = 1.0
  elif sym:                  g = 0.0 if a == b else 1.0
  else:                      g = abs(a - b)
  return g * g


def _ydist(goals: list, row: list) -> float:
  "How far ROW's goals, taken together, are from the ideal; a missing goal is worst."
  gaps = [abs(ezr.norm(m, row[at]) - m.goal) if _num(row[at]) else 1.0
          for at, m in goals]
  return ezr.dist(gaps, len(gaps))


def fix(train_csv: str, row, frozen=()) -> dict:
  """The inputs of ROW (a bad player, goals removed), changed as little as
  possible so that its predicted goals fall in the best quarter of
  TRAIN_CSV.  Columns named in FROZEN never change.  Changes are searched
  exhaustively for one and for two inputs (two over the columns that help
  most alone), then greedily; if no good player is reached, the closest
  one found is returned."""
  data = _prepare(train_csv)
  if data is None: return {}
  names, rows, goals, inputs, kept = data
  base = _full(names, kept, inputs, row)
  if not inputs: return {}
  if not rows or not goals:
    return {names[at]: base[at] for at, _ in inputs}

  if isinstance(frozen, str): frozen = (frozen,)
  pinned = set()
  for f in frozen:
    s = str(f).strip()
    pinned |= {s, s.rstrip("+-!")}
  movable = [j for j, (at, _) in enumerate(inputs)
             if names[at] not in pinned
             and names[at].rstrip("+-!") not in pinned]

  n    = len(rows)
  k    = min(n, max(5, int(math.sqrt(n))))
  p    = len(inputs)
  sym  = [m is None for _, m in inputs]
  E    = [[_enc(m, r[at]) for at, m in inputs] for r in rows]
  bq   = [_enc(m, base[at]) for at, m in inputs]
  vals = [_levels(rows, at, m is not None) for at, m in inputs]
  enc  = [[_enc(m, v) for v in vals[j]]
          for j, (_, m) in enumerate(inputs)]
  opts = {j: [t for t, v in enumerate(vals[j])
              if v != base[inputs[j][0]]]
          for j in movable}

  # Distances from every training row to the bad row, and to each candidate.
  base_sq  = [[_g2(sym[j], bq[j], E[i][j]) for i in range(n)]
              for j in range(p)]
  base_tot = [sum(col[i] for col in base_sq) for i in range(n)]
  delta    = {}
  for j in movable:
    for t in opts[j]:
      cand = enc[j][t]
      delta[(j, t)] = [_g2(sym[j], cand, E[i][j]) - base_sq[j][i]
                       for i in range(n)]

  gall = [[r[at] for r in rows if _num(r[at])] for at, _ in goals]
  cut  = _quantile([_ydist(goals, r) for r in rows], _GOOD_Q)

  def answer(changes: dict) -> dict:
    "The inputs of the bad row, with CHANGES (input position -> value index) applied."
    return {names[at]: vals[j][changes[j]] if j in changes else base[at]
            for j, (at, _) in enumerate(inputs)}

  def predict(changes: dict) -> float:
    "Predicted distance from the ideal, with CHANGES applied to the bad row."
    tot = list(base_tot)
    for j, t in changes.items():
      tot = [a + b for a, b in zip(tot, delta[(j, t)])]
    order = sorted(range(n), key=lambda i: (tot[i], i))[:k]
    gaps  = []
    for (at, m), allv in zip(goals, gall):
      vs = [rows[i][at] for i in order if _num(rows[i][at])] or allv
      if not vs:
        gaps.append(1.0)
        continue
      est = _quantile(vs, 0.5)
      gaps.append(abs(ezr.norm(m, est) - m.goal))
    return ezr.dist(gaps, len(gaps))

  best_seen = [math.inf, {}]

  def score(changes: dict) -> float:
    "predict, remembering the closest candidate seen so far."
    y = predict(changes)
    if y < best_seen[0]: best_seen[:] = [y, dict(changes)]
    return y

  # No change needed.
  if score({}) <= cut: return answer({})

  # One change: try every candidate value of every movable input.
  best1, col_best = None, {}
  for j in movable:
    for t in opts[j]:
      y = score({j: t})
      col_best[j] = min(col_best.get(j, math.inf), y)
      if best1 is None or y < best1[0]: best1 = (y, {j: t})
  if best1 is not None and best1[0] <= cut: return answer(best1[1])

  # Two changes: only among the columns that help most on their own.
  top   = sorted(col_best, key=lambda j: (col_best[j], j))[:_PAIR_COLS]
  best2 = None
  for a, b in combinations(sorted(top), 2):
    for ta, tb in product(opts[a], opts[b]):
      ch = {a: ta, b: tb}
      y  = score(ch)
      if best2 is None or y < best2[0]: best2 = (y, ch)
  if best2 is not None and best2[0] <= cut: return answer(best2[1])

  # Greedy: keep taking the single change that helps most.
  cur, cur_y = {}, predict({})
  for _ in range(_STEPS):
    if cur_y <= cut: break
    step = None
    for j in movable:
      for t in opts[j]:
        trial = dict(cur)
        trial[j] = t
        y = score(trial)
        if step is None or y < step[0]: step = (y, trial)
    if step is None or step[0] >= cur_y: break
    cur_y, cur = step
  if cur_y <= cut: return answer(cur)
  return answer(best_seen[1])

