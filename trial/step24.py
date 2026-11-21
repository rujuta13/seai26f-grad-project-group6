# step24.py
"""Step 24: flow.  Welcome offers for new sign-ups as they arrive, and a
warning when this month's sign-ups stop looking like the usual customers.

A sign-up's goals (purchases, visits, response, complaints) are not known
when it arrives, so stream() decides from its input columns alone.  It
predicts each goal from the nearest customers it already knows, and keeps
the sign-up if that prediction is in the best quarter seen so far.  drift()
also reads only input columns.

Identifier and text columns are left out, as in steps 11 and 16.  Goals are
the numeric columns named with a trailing "+" or "-"; every other kept column
is an input.

Uses only the standard library plus ezr.
"""
import bisect
import math
import random
import ezr

_GOOD_Q = 0.25   # best quarter of customers counts as good
_KEEP_Q = 0.25   # keep a sign-up if its predicted goals are in the best quarter seen
_SAMPLE = 100    # training customers used as the novelty reference
_WIN    = 30     # sign-ups per drift block
_MIN    = 10     # smallest trailing block worth testing
_CLIFF  = 0.33   # strict Cliff's delta threshold (a medium effect)
_KS     = 1.63   # Kolmogorov-Smirnov critical value at about the 99% level
_SHARE  = 0.4    # share of numeric inputs that must shift for a block to differ
_DROP   = {"ID", "num", "Rank", "Name", "Nation", "League", "Team", "url"}


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


def _load(path: str):
  "Read PATH: (header names, full-width rows with cleaned cells), or None if empty."
  raw = list(ezr.csv(path))
  if not raw: return None
  names = [str(h).strip() for h in raw[0]]
  rows  = []
  for r in raw[1:]:
    r = [_clean(v) for v in r][:len(names)]
    rows.append(r + ["?"] * (len(names) - len(r)))
  return names, rows


def _is_num(rows: list, at: int) -> bool:
  "True if most known cells of column AT are numbers."
  known = [r[at] for r in rows if r[at] != "?"]
  nums  = sum(1 for v in known if _num(v))
  return nums > 0 and 2 * nums >= len(known)


def _model(names: list, rows: list, at: int) -> ezr.Num:
  "An ezr.Num summarising the numeric cells of column AT; its goal sign comes from its name."
  return ezr.adds([r[at] for r in rows if _num(r[at])],
                  ezr.Num(names[at], at))


def _layout(names: list, rows: list):
  """(goals, inputs, kept).  GOALS are (position, Num) for numeric goals;
  INPUTS are (position, Num) or (position, None) for symbolic inputs; KEPT
  lists every usable column position, goals included."""
  goals, inputs, kept = [], [], []
  for at, nm in enumerate(names):
    if not nm or nm in _DROP or nm.endswith("X"): continue
    kept.append(at)
    numeric = _is_num(rows, at)
    if nm[-1] in "+-":
      if numeric: goals.append((at, _model(names, rows, at)))
    else:
      inputs.append((at, _model(names, rows, at) if numeric else None))
  return goals, inputs, kept


def _enc(m, v):
  "An input cell for distance: None if unknown, symbols as-is, numbers scaled to 0..1."
  if m is None: return None if v == "?" else v
  return ezr.norm(m, v) if _num(v) else None


def _gap(sym: bool, a, b) -> float:
  "Gap between two encoded cells of one input; unknown is far."
  if a is None or b is None: return 1.0
  if sym: return 0.0 if a == b else 1.0
  return abs(a - b)


def _dist(sym: list[bool], a: list, b: list) -> float:
  "Root mean square gap between two encoded input rows, as ezr.xdist."
  if not sym: return 0.0
  return math.sqrt(sum(_gap(f, x, y) ** 2
                       for f, x, y in zip(sym, a, b)) / len(sym))


def _ydist(goals: list, vals: list) -> float:
  "RMS distance of goal values VALS (aligned with GOALS) from their ideal; unknown is worst."
  gaps = [abs(ezr.norm(m, v) - m.goal) if _num(v) else 1.0
          for (_, m), v in zip(goals, vals)]
  return ezr.dist(gaps, len(gaps)) if gaps else 0.0


def _pool(goals: list, G: list) -> list[float]:
  "Sorted goal distances of the fully known rows of G (every row, if none is fully known)."
  full = [g for g in G if all(_num(v) for v in g)] or G
  return sorted(_ydist(goals, g) for g in full)


def _same(xs, ys) -> bool:
  "True if XS and YS look alike at strict thresholds; cliffs and ks must both agree."
  xs, ys = sorted(xs), sorted(ys)
  if not xs or not ys: return True            # no evidence of a shift
  return ezr.cliffs(xs, ys, d=_CLIFF) and ezr.ks(xs, ys, a=_KS)


def _setup(train_csv: str):
  "What the training customers look like: layout, encoded inputs and raw goal cells."
  loaded = _load(train_csv)
  if loaded is None: return None
  names, rows = loaded
  if not rows: return None
  goals, inputs, kept = _layout(names, rows)
  sym = [m is None for _, m in inputs]
  E   = [[_enc(m, r[at]) for at, m in inputs] for r in rows]
  G   = [[r[at] for at, _ in goals] for r in rows]
  return ezr.o(names=names, kept=kept, goals=goals, inputs=inputs,
               sym=sym, E=E, G=G)


def _full(M, row) -> list:
  "Lay ROW out full-width in header order, '?' where missing; cells are cleaned."
  names = M.names
  q     = ["?"] * len(names)
  if isinstance(row, dict):                    # {column name: value}
    stems = {str(k).strip().rstrip("+-!"): v for k, v in row.items()}
    for at, nm in enumerate(names):
      stem = nm.rstrip("+-!")
      if nm in row:           q[at] = row[nm]
      elif stem in stems:     q[at] = stems[stem]
  else:
    seq = list(row)
    if len(seq) == len(names):      targets = range(len(names))
    elif len(seq) == len(M.kept):   targets = M.kept
    else:                           targets = [at for at, _ in M.inputs]
    for at, v in zip(targets, seq): q[at] = v
  return [_clean(v) for v in q]


def _predict(M, kE: list, kG: list, e: list) -> list:
  "Median of each goal among the nearest known customers to encoded inputs E; '?' if none know it."
  n     = len(kE)
  k     = min(n, max(5, int(math.sqrt(n))))
  order = sorted(range(n), key=lambda i: (_dist(M.sym, e, kE[i]), i))[:k]
  est   = []
  for j in range(len(M.goals)):
    vs = [kG[i][j] for i in order if _num(kG[i][j])]
    if not vs: vs = [g[j] for g in kG if _num(g[j])]
    est.append(_quantile(vs, 0.5) if vs else "?")
  return est


def stream(train_csv: str, rows) -> list[bool]:
  """For each row in ROWS, in order: True if its predicted goals are in the
  best quarter seen so far, else False.  The decision reads only the row's
  input columns, since its goals are not yet known.  Its goal cells are
  learned from only after the decision, as they are for the rows before it."""
  rows = list(rows)
  M    = _setup(train_csv)
  if M is None or not M.goals: return [False] * len(rows)
  kE   = [list(e) for e in M.E]
  kG   = [list(g) for g in M.G]
  pool = _pool(M.goals, kG)                       # what "good" looks like
  keep = []
  for raw in rows:
    q   = _full(M, raw)
    e   = [_enc(m, q[at]) for at, m in M.inputs]
    cut = pool[int(_KEEP_Q * (len(pool) - 1))]    # decide before learning
    est = _predict(M, kE, kG, e)
    keep.append(_ydist(M.goals, est) <= cut)
    g = [q[at] for at, _ in M.goals]              # learn from it afterwards
    kE.append(e)
    kG.append(g)
    if all(_num(v) for v in g):
      bisect.insort(pool, _ydist(M.goals, g))
  return keep


def drift(train_csv: str, rows) -> int | None:
  """Index into ROWS of the row at which the sign-ups first stop looking like
  TRAIN_CSV, or None.  Only input columns are read.  Rows are tested in blocks
  of _WIN.  A block differs if its novelty (distance to the nearest training
  customer) differs from the training customers' own novelty, or if at least
  _SHARE of its numeric inputs shift.  Warn at the end of the second differing
  block in a row, so a single noisy block does not raise an alarm."""
  rows = list(rows)
  M    = _setup(train_csv)
  if M is None or not rows or not M.inputs: return None
  E, sym, n = M.E, M.sym, len(M.E)
  num_at    = [j for j, s in enumerate(sym) if not s]

  # Training customers' own novelty: each sampled one's nearest other customer.
  rng   = random.Random(ezr.the.Seed)
  ref_f = []
  for i in rng.sample(range(n), min(_SAMPLE, n)):
    d = [_dist(sym, E[i], E[j]) for j in range(n) if j != i]
    if d: ref_f.append(min(d))
  if not ref_f: ref_f = [0.0]
  col_ref = {j: sorted(x for x in (E[i][j] for i in range(n)) if x is not None)
             for j in num_at}

  Q, f = [], []
  for raw in rows:
    q = _full(M, raw)
    e = [_enc(m, q[at]) for at, m in M.inputs]
    Q.append(e)
    f.append(min(_dist(sym, e, t) for t in E))

  need = max(3, math.ceil(_SHARE * len(num_at)))
  run  = 0
  for lo in range(0, len(rows), _WIN):
    hi = min(lo + _WIN, len(rows))
    if hi - lo < _MIN: break
    shifted = 0
    for j in num_at:
      xs = [Q[i][j] for i in range(lo, hi) if Q[i][j] is not None]
      if not _same(xs, col_ref[j]): shifted += 1
    differs = (not _same(f[lo:hi], ref_f)) or shifted >= need
    run = run + 1 if differs else 0
    if run >= 2: return hi - 1
  return None

