# step25.py
"""Step 25: blame.  Which input columns go with the customers who never
respond to campaigns and the ones who complain?

Customers are read as in step21: identifier and text columns are left out,
goals are the numeric columns named with a trailing "+" (higher is better)
or "-" (lower is better), and every other kept column is an input.  A bad
customer is one whose goals, taken together, are in the worst quarter; a
good one is in the best quarter.

Uses only the standard library plus ezr.
"""
import math
from collections import Counter
import ezr

_GOOD_Q = 0.25   # best quarter of customers counts as good
_BAD_Q  = 0.75   # worst quarter counts as bad
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


def _load(train_csv: str):
  "Read TRAIN_CSV: (header names, full-width rows with cleaned cells), or None if empty."
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
  for at, nm in enumerate(names):
    if not nm or nm in _DROP or nm.endswith("X"): continue
    numeric = _is_num(rows, at)
    if nm[-1] in "+-":
      if numeric: goals.append((at, _model(names, rows, at)))
    else:
      inputs.append((at, _model(names, rows, at) if numeric else None))
  return names, rows, goals, inputs


def _ydist(goals: list, row: list) -> float:
  "How far ROW's goals, taken together, are from the ideal; a missing goal is worst."
  gaps = [abs(ezr.norm(m, row[at]) - m.goal) if _num(row[at]) else 1.0
          for at, m in goals]
  return math.sqrt(sum(g * g for g in gaps) / len(gaps)) if gaps else 0.0


def _shares(rs: list, at: int) -> dict:
  "Share of the known cells of column AT in RS, per value seen."
  vals = [r[at] for r in rs if r[at] != "?"]
  if not vals: return {}
  n = len(vals)
  return {v: k / n for v, k in Counter(vals).items()}


def _score(names: list, at: int, m, bad: list, good: list) -> float:
  "How strongly input column AT separates bad rows from good rows."
  if m is not None:
    vb = [ezr.norm(m, r[at]) for r in bad  if _num(r[at])]
    vg = [ezr.norm(m, r[at]) for r in good if _num(r[at])]
    if not vb or not vg: return 0.0
    return abs(sum(vb) / len(vb) - sum(vg) / len(vg))
  sb, sg = _shares(bad, at), _shares(good, at)
  values = set(sb) | set(sg)
  if not values: return 0.0
  return max(abs(sb.get(v, 0.0) - sg.get(v, 0.0)) for v in values)


def blame(train_csv: str) -> list[str]:
  """Input column names, most to blame for bad rows (bad on the goals) first.
  A row is bad if its goals together are in the worst quarter, and good if
  in the best quarter.  A numeric input scores the gap between its mean
  (scaled to 0..1) in bad and good rows.  A symbolic input scores the
  largest gap, over its values, between how often that value appears in
  bad rows and in good rows.  Ties keep file order."""
  data = _prepare(train_csv)
  if data is None: return []
  names, rows, goals, inputs = data
  if not inputs: return []
  if not rows or not goals:
    return [names[at] for at, _ in inputs]

  ys   = [_ydist(goals, r) for r in rows]
  hi   = _quantile(ys, _BAD_Q)
  lo   = _quantile(ys, _GOOD_Q)
  bad  = [r for r, y in zip(rows, ys) if y >= hi]
  good = [r for r, y in zip(rows, ys) if y <= lo]
  if not bad or not good:
    return [names[at] for at, _ in inputs]

  scored = [(_score(names, at, m, bad, good), i, names[at])
            for i, (at, m) in enumerate(inputs)]
  scored.sort(key=lambda t: (-t[0], t[1]))   # most blame first; ties by file order
  return [name for _, _, name in scored]

