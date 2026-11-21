# step26.py
"""Step 26: justify.  Explain to the sales director why this group of
customers should get the next campaign, and say in a sentence or two why
one customer was rated low value, with reasons that can be acted on.

Customers are read as in step21: identifier and text columns are left
out, goals are the numeric columns named with a trailing "+" (higher is
better) or "-" (lower is better), and every other kept column is an input.
A column is numeric if most of its known cells are numbers; text cells in
a numeric column count as unknown.

Uses only the standard library plus ezr.
"""
import math
from collections import Counter
import ezr

_GOOD_Q = 0.25   # best quarter of customers counts as the target group
_BAD_Q  = 0.75   # worst quarter counts as low value
_SHIFT  = 0.10   # an input must shift by at least 10 points to be reported
_TOP    = 5      # how many inputs to explain
_DROP   = {"ID", "num", "Rank", "Name", "Nation", "League", "Team", "url"}
_NOT_ENOUGH = "There is not enough training data to explain anything."


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


def _avg(xs: list[float]):
  "Mean of XS, or None if empty."
  return sum(xs) / len(xs) if xs else None


def _fmt(x) -> str:
  "A number for display, or 'n/a'."
  return "n/a" if x is None else ezr.say(x)


def _way(m: ezr.Num) -> str:
  "Plain words for which direction of a goal is better."
  return "higher is better" if m.goal else "lower is better"


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
  """Load TRAIN_CSV: (names, rows, goals, inputs, kept).  GOALS are (position, Num)
  for numeric goals; INPUTS are (position, Num) or (position, None) when the
  input is symbolic; KEPT lists every usable column position, goals included."""
  loaded = _load(train_csv)
  if loaded is None: return None
  names, rows = loaded
  kept, goals, inputs = [], [], []
  for at, nm in enumerate(names):
    if not nm or nm in _DROP or nm.endswith("X"): continue
    kept.append(at)
    numeric = _is_num(rows, at)
    if nm[-1] in "+-":
      if numeric: goals.append((at, _model(names, rows, at)))
    else:
      inputs.append((at, _model(names, rows, at) if numeric else None))
  return names, rows, goals, inputs, kept


def _full(names: list, kept: list, row) -> list:
  "Lay ROW out full-width in header order, '?' where missing; cells are cleaned."
  q = ["?"] * len(names)
  if isinstance(row, dict):                    # {column name: value}
    by_stem = {str(k).strip().rstrip("+-!"): v for k, v in row.items()}
    for at, name in enumerate(names):
      stem = name.rstrip("+-!")
      if name in row:          q[at] = row[name]
      elif stem in by_stem:    q[at] = by_stem[stem]
  elif len(row) == len(names):                 # already full-width
    q = list(row)
  else:                                        # positional, over kept columns
    for at, v in zip(kept, row): q[at] = v
  return [_clean(v) for v in q]


def _gaps(goals: list, row: list) -> list[float]:
  "Per goal: distance from its ideal (0 is best); a missing goal is the worst, 1."
  return [abs(ezr.norm(m, row[at]) - m.goal) if _num(row[at]) else 1.0
          for at, m in goals]


def _ydist(goals: list, row: list) -> float:
  "How far ROW's goals, taken together, are from the ideal."
  gs = _gaps(goals, row)
  return ezr.dist(gs, len(gs)) if gs else 0.0


def _shares(rs: list, at: int) -> dict:
  "Share of the known cells of column AT in RS, per value seen."
  vals = [r[at] for r in rs if r[at] != "?"]
  if not vals: return {}
  n = len(vals)
  return {v: k / n for v, k in Counter(vals).items()}


def explain(train_csv: str) -> str:
  "Plain-words reasons, checkable on the training data, why the best-quarter customers are the ones to target."
  data = _prepare(train_csv)
  if data is None: return _NOT_ENOUGH
  names, rows, goals, inputs, _kept = data
  if not rows or not goals or not inputs: return _NOT_ENOUGH

  ys   = [_ydist(goals, r) for r in rows]
  cut  = _quantile(ys, _GOOD_Q)
  good = [r for r, y in zip(rows, ys) if y <= cut]

  avg_parts = []
  for at, _ in goals:
    gv = _avg([r[at] for r in good if _num(r[at])])
    av = _avg([r[at] for r in rows if _num(r[at])])
    avg_parts.append(f"{names[at]} {_fmt(gv)} vs {_fmt(av)}")

  out = [
    f"Target group: the best quarter of {len(rows)} training customers on all "
    f"goals together, {len(good)} customers, with distance from the ideal at "
    f"most {cut:.2f}.",
    "Goals the campaign should move: "
    + ", ".join(f"{names[at]} ({_way(m)})" for at, m in goals) + ".",
    "Average goals, target group vs all customers: "
    + "; ".join(avg_parts) + ".",
  ]

  findings = []                                       # (size, sentence)
  for at, m in inputs:
    name = names[at]
    if m is not None:
      gv_raw = [r[at] for r in good if _num(r[at])]
      av_raw = [r[at] for r in rows if _num(r[at])]
      if not gv_raw or not av_raw: continue
      mg = sum(ezr.norm(m, v) for v in gv_raw) / len(gv_raw)
      ma = sum(ezr.norm(m, v) for v in av_raw) / len(av_raw)
      if abs(mg - ma) >= _SHIFT:
        way = "higher" if mg > ma else "lower"
        findings.append((abs(mg - ma),
                         f"{name} is {way} in the target group: average "
                         f"{_fmt(_avg(gv_raw))} vs {_fmt(_avg(av_raw))} "
                         f"across all customers."))
    else:
      sg, sa = _shares(good, at), _shares(rows, at)
      if not sa: continue
      v     = max(sorted(set(sa) | set(sg), key=str),
                  key=lambda k: abs(sg.get(k, 0) - sa.get(k, 0)))
      shift = sg.get(v, 0) - sa.get(v, 0)
      if abs(shift) >= _SHIFT:
        findings.append((abs(shift),
                         f"{name}={ezr.say(v)} appears in {sg.get(v, 0):.0%} "
                         f"of target-group customers but {sa.get(v, 0):.0%} "
                         f"of all customers."))
  findings.sort(key=lambda t: -t[0])                  # stable: ties keep file order

  if findings:
    out.append("Inputs that separate the target group from the rest, largest shift first:")
    out += [f"  {i}. {s}" for i, (_, s) in enumerate(findings[:_TOP], 1)]
  else:
    out.append(f"No input shifts by {_SHIFT:.0%} or more between the target group "
               "and the rest, so there is no clear input reason in the data.")
  out.append("These are associations in the training data, not proof of cause. "
             "Before a full rollout, send the campaign to a random sample of the "
             "target group and compare its response with a random sample of other "
             "customers of the same size.")
  return "\n".join(out)


def why(train_csv: str, row) -> str:
  "Two sentences: is this customer (goals included) high, middle, or low value, and which goal decides it."
  data = _prepare(train_csv)
  if data is None:
    return "Cannot rate this customer: the training data has no goals."
  names, rows, goals, _inputs, kept = data
  if not rows or not goals:
    return "Cannot rate this customer: the training data has no goals."

  if isinstance(row, int) and not isinstance(row, bool):
    if not 0 <= row < len(rows):
      return f"Cannot rate this customer: there is no training row {row}."
    q = rows[row]                                     # a training row, by index
  else:
    q = _full(names, kept, row)
  if not any(_num(q[at]) for at, _ in goals):
    return "Cannot rate this customer: its goal values are missing."

  ys = [_ydist(goals, r) for r in rows]
  lo = _quantile(ys, _GOOD_Q)
  hi = _quantile(ys, _BAD_Q)
  y  = _ydist(goals, q)
  if y <= lo:   verdict, tier = "high value", "best quarter"
  elif y >= hi: verdict, tier = "low value", "worst quarter"
  else:         verdict, tier = "middle value", "middle half"

  gaps  = _gaps(goals, q)
  k     = max(range(len(goals)), key=lambda i: gaps[i])   # first of ties
  at, m = goals[k]
  v     = q[at]
  if _num(v):
    good = [r for r, yy in zip(rows, ys) if yy <= lo]
    gavg = _avg([r[at] for r in good if _num(r[at])])
    aavg = _avg([r[at] for r in rows if _num(r[at])])
    s2 = (f"The goal furthest from its ideal is {names[at]} at {ezr.say(v)} "
          f"({_way(m)}; the best quarter averages {_fmt(gavg)} and all "
          f"customers {_fmt(aavg)}).")
  else:
    s2 = (f"The goal furthest from its ideal is {names[at]}, "
          "which is missing for this customer.")

  s1 = (f"Rated {verdict}: its goals put it in the {tier} of the "
        f"{len(rows)} training customers (distance to the ideal {y:.2f}; "
        f"best-quarter cutoff {lo:.2f}, worst-quarter cutoff {hi:.2f}).")
  return s1 + " " + s2

