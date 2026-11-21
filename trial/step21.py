# step21.py
"""Step 21: remember.  Group customers into the kinds of shopper they really
are, and find the best customers: busy across channels, loyal, no complaints.

Identifier and text columns are left out, as in steps 11 and 16.  Goals are
the numeric columns named with a trailing "+" (higher is better) or "-"
(lower is better); every other kept column is an input.  A column is numeric
if most of its known cells are numbers; text cells in a numeric column count
as unknown.

Uses only the standard library plus ezr.
"""
import math
import random
from collections import Counter
import ezr

_K      = 5                                   # how many shopper kinds to look for
_ROUNDS = 20                                  # most passes of the family search
_CUT    = 0.05                                # best-on-all-goals: at most this share
_DROP   = {"ID", "num", "Rank", "Name", "Nation", "League", "Team", "url"}


def _num(v) -> bool:
  "True if V is a usable number (not text, not missing, not a bool)."
  return isinstance(v, (int, float)) and not isinstance(v, bool)


def _clean(v):
  "A cell as ezr reads it: strings are cast, blanks and non-finite numbers become '?'."
  if v is None: return "?"
  if isinstance(v, str):
    s = v.strip()
    v = ezr.atom(s) if s else "?"
  if isinstance(v, float) and not math.isfinite(v): return "?"
  return v


def _load(train_csv: str):
  "Read TRAIN_CSV: (header names, full-width rows with '?' where missing)."
  raw = list(ezr.csv(train_csv))
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


def _prepare(train_csv: str):
  """Load TRAIN_CSV: (names, rows, goals, inputs).  GOALS are (position, Num)
  for numeric goals; INPUTS are (position, Num) or (position, None) when
  the input is symbolic."""
  loaded = _load(train_csv)
  if loaded is None: return None
  names, rows = loaded
  goals, inputs = [], []
  for at, name in enumerate(names):
    if not name or name in _DROP or name.endswith("X"): continue
    numeric = _is_num(rows, at)
    if name[-1] in "+-":
      if numeric: goals.append((at, _model(names, rows, at)))
    else:
      inputs.append((at, _model(names, rows, at) if numeric else None))
  return names, rows, goals, inputs


def _enc(m, v):
  "An input cell for distance: None if unknown, symbols as-is, numbers scaled to 0..1."
  if isinstance(v, str) and v == "?": return None
  if m is None: return v
  return ezr.norm(m, v) if _num(v) else None


def _dist(sym: list[bool], a: list, b: list) -> float:
  "Root mean square gap between two encoded input rows, as ezr.xdist."
  if not sym: return 0.0
  s = 0.0
  for f, x, y in zip(sym, a, b):
    if x is None or y is None: g = 1.0
    elif f:                    g = 0.0 if x == y else 1.0
    else:                      g = abs(x - y)
    s += g * g
  return math.sqrt(s / len(sym))


def _centre(members: list[list], sym: list[bool]) -> list:
  "The middle of encoded rows: mean for numbers, most common for symbols; None if unknown."
  out = []
  for j, f in enumerate(sym):
    vs = [m[j] for m in members if m[j] is not None]
    if not vs:   out.append(None)
    elif f:      out.append(Counter(vs).most_common(1)[0][0])
    else:        out.append(sum(vs) / len(vs))
  return out


def _gap(m: ezr.Num, v) -> float:
  "Distance of one goal cell from its ideal (0 is best); a missing goal is the worst, 1."
  return abs(ezr.norm(m, v) - m.goal) if _num(v) else 1.0


def families(train_csv: str) -> list[int]:
  "One family label per data row of TRAIN_CSV, in file order."
  data = _prepare(train_csv)
  if data is None: return []
  _names, rows, _goals, inputs = data
  n = len(rows)
  if n == 0: return []
  if not inputs: return [0] * n
  sym   = [m is None for _, m in inputs]
  E     = [[_enc(m, r[at]) for at, m in inputs] for r in rows]
  k     = min(_K, n)
  rng   = random.Random(ezr.the.Seed)        # local rng: no global side effects
  cents = [E[i] for i in rng.sample(range(n), k)]
  labels = None
  for _ in range(_ROUNDS):
    new = [min(range(k), key=lambda j: (_dist(sym, e, cents[j]), j))
           for e in E]
    if new == labels: break                  # assignments are stable
    labels = new
    for j in range(k):
      members = [e for e, l in zip(E, labels) if l == j]
      if members:                            # keep old centre if empty
        cents[j] = _centre(members, sym)
  return labels


def best(train_csv: str) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows."
  data = _prepare(train_csv)
  if data is None: return []
  _names, rows, goals, _inputs = data
  n   = len(rows)
  cap = int(_CUT * n)
  if cap <= 0 or not goals: return []

  # Per goal: distance from the ideal, smaller is better; unknown is worst.
  G   = [[_gap(m, r[at]) for at, m in goals] for r in rows]
  rms = [math.sqrt(sum(g * g for g in gs) / len(gs)) for gs in G]

  # Lexicographic order puts any dominating row before the rows it dominates,
  # so each row needs checking only against the front found so far.
  order = sorted(range(n), key=lambda i: (G[i], i))
  front: list[int] = []
  for i in order:
    gi = G[i]
    if not any(all(a <= b for a, b in zip(G[j], gi)) and G[j] != gi
               for j in front):
      front.append(i)

  # Keep the closest to the ideal, capped at 5%; return in file order.
  front.sort(key=lambda i: (rms[i], i))
  return sorted(front[:cap])

