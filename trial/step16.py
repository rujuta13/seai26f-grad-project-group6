# step16.py
"""Step 16: justify.  Explain to a manager why a player is worth signing,
and say in a sentence or two why one player was rated low, with reasons
a skeptical expert can check on video.

Identifier and text columns (num, Rank, Name, Nation, League, Team, url)
are left out, as in steps 11 and 15.  Goals are the columns whose names end
in "+" (higher is better) or "-" (lower is better); every other kept column
is an input.  A column is numeric if most of its known cells are numbers;
text cells in a numeric column count as unknown.

Uses only the standard library plus ezr.
"""
import math
from collections import Counter
import ezr

_GOOD_Q = 0.25   # best quarter of training players counts as good
_BAD_Q  = 0.75   # worst quarter counts as bad
_SHIFT  = 0.10   # an input must shift by at least 10 points to be reported
_TOP    = 5      # how many inputs to explain
_DROP   = {"num", "Rank", "Name", "Nation", "League", "Team", "url"}
_NOT_ENOUGH = "There is not enough training data to explain anything."


def _clean(v):
  "A cell as ezr reads it: strings are cast, blanks and None become '?'."
  if v is None: return "?"
  if isinstance(v, str): return ezr.atom(v) if v.strip() else "?"
  return v


def _num(v) -> bool:
  "True if V is a usable number (not text, not missing, not a bool)."
  return isinstance(v, (int, float)) and not isinstance(v, bool)


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
  """Load TRAIN_CSV: (names, rows, goals, inputs).  GOALS and INPUTS are
  lists of (position, model); an input model is None if it is symbolic."""
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


def _quantile(xs: list[float], p: float) -> float:
  "The p-th quantile of XS, linearly interpolated.  XS must be non-empty."
  xs     = sorted(xs)
  i      = p * (len(xs) - 1)
  lo, hi = math.floor(i), math.ceil(i)
  return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def _gaps(goals: list, row: list) -> list[float]:
  "Per goal: distance from its ideal (0 is best); a missing goal is the worst, 1."
  return [abs(ezr.norm(m, row[at]) - m.goal) if _num(row[at]) else 1.0
          for at, m in goals]


def _ydist(goals: list, row: list) -> float:
  "How far ROW's goals, taken together, are from the ideal."
  gs = _gaps(goals, row)
  return ezr.dist(gs, len(gs))


def _way(m: ezr.Num) -> str:
  "Plain words for which direction of a goal is better."
  return "higher is better" if m.goal else "lower is better"


def _avg(xs: list[float]):
  "Mean of XS, or None if empty."
  return sum(xs) / len(xs) if xs else None


def _fmt(x) -> str:
  "A number for display, or 'n/a'."
  return "n/a" if x is None else ezr.say(x)


def _shares(rs: list, at: int) -> dict:
  "Share of the known cells of column AT in RS, per value seen."
  vals = [r[at] for r in rs if r[at] != "?"]
  return {v: k / len(vals) for v, k in Counter(vals).items()} if vals else {}


def _full(names: list, row) -> list:
  "Lay ROW out full-width in header order, '?' where missing; strings are cast."
  q = ["?"] * len(names)
  if isinstance(row, dict):                    # {column name: value}
    by_stem = {str(k).strip().rstrip("+-!"): v for k, v in row.items()}
    for at, name in enumerate(names):
      stem = name.rstrip("+-!")
      if name in row: q[at] = row[name]
      elif stem in by_stem: q[at] = by_stem[stem]
  elif len(row) == len(names):                 # already full-width
    q = list(row)
  else:                                        # positional, over kept columns
    kept = [at for at, name in enumerate(names)
            if name and name not in _DROP and not name.endswith("X")]
    for at, v in zip(kept, row): q[at] = v
  return [_clean(v) for v in q]


def explain(train_csv: str) -> str:
  "Plain-words reasons, checkable on the training data, for what makes players good."
  data = _prepare(train_csv)
  if data is None: return _NOT_ENOUGH
  names, rows, goals, inputs = data
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
    f"Training data: {len(rows)} players. Good means closest to the ideal "
    f"on all goals together (best quarter): {len(good)} players, "
    f"distance to the ideal at most {cut:.2f}.",
    "Goals: " + ", ".join(f"{names[at]} ({_way(m)})"
                          for at, m in goals) + ".",
    "Average goals, best quarter vs all players: "
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
                         f"{name} is {way} in the best quarter: median "
                         f"{ezr.say(_quantile(gv_raw, 0.5))} vs "
                         f"{ezr.say(_quantile(av_raw, 0.5))} across all players."))
    else:
      sg, sa = _shares(good, at), _shares(rows, at)
      if not sa: continue
      v     = max(sorted(set(sa) | set(sg), key=str),
                  key=lambda k: abs(sg.get(k, 0) - sa.get(k, 0)))
      shift = sg.get(v, 0) - sa.get(v, 0)
      if abs(shift) >= _SHIFT:
        findings.append((abs(shift),
                         f"{name}={ezr.say(v)} appears in {sg.get(v, 0):.0%} "
                         f"of best-quarter players but {sa.get(v, 0):.0%} "
                         f"of all players."))
  findings.sort(key=lambda t: -t[0])                  # stable: ties keep file order

  if findings:
    out.append("Inputs that separate the best quarter from the rest, largest shift first:")
    out += [f"  {i}. {s}" for i, (_, s) in enumerate(findings[:_TOP], 1)]
  else:
    out.append(f"No input shifts by {_SHIFT:.0%} or more between the best quarter "
               "and the rest, so there is no clear input reason in the data.")
  out.append("These are associations in the training data, not proof of cause. "
             "Check each one on video, comparing the player with a similar player "
             "who differs in that one input.")
  return "\n".join(out)


def why(train_csv: str, row) -> str:
  "One or two sentences: is this row (goals included) good or bad, and which goal decides it."
  data = _prepare(train_csv)
  if data is None: return "Cannot rate this player: the training data has no goals."
  names, rows, goals, _ = data
  if not rows or not goals:
    return "Cannot rate this player: the training data has no goals."

  if isinstance(row, int) and not isinstance(row, bool):
    q = rows[row]                                     # a training row, by index
  else:
    q = _full(names, row)
  if not any(_num(q[at]) for at, _ in goals):
    return "Cannot rate this player: its goal values are missing."

  ys = [_ydist(goals, r) for r in rows]
  lo = _quantile(ys, _GOOD_Q)
  hi = _quantile(ys, _BAD_Q)
  y  = _ydist(goals, q)
  if y <= lo:   verdict, tier = "good", "best quarter"
  elif y >= hi: verdict, tier = "bad", "worst quarter"
  else:         verdict, tier = "middling", "middle half"

  gaps = _gaps(goals, q)
  k    = max(range(len(goals)), key=lambda i: gaps[i])   # first of ties
  at, m = goals[k]
  v    = q[at]
  if _num(v):
    med = _quantile([r[at] for r in rows if _num(r[at])], 0.5)
    s2  = (f"The goal furthest from its ideal is {names[at]} at {ezr.say(v)} "
           f"({_way(m)}; training median {ezr.say(med)}).")
  else:
    s2  = (f"The goal furthest from its ideal is {names[at]}, "
           "which is missing from this row.")

  s1 = (f"Rated {verdict}: its goals put it in the {tier} of the "
        f"{len(rows)} training players (distance to the ideal {y:.2f}; "
        f"best-quarter cutoff {lo:.2f}, worst-quarter cutoff {hi:.2f}).")
  return s1 + " " + s2

