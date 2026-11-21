# step6.py
"""Step 6: justify.  Explain the recommended flag set to a skeptic, and
say in a sentence or two why one build was rated good or bad.

Uses only the standard library plus ezr.
"""
import math
import ezr

_GOOD_Q = 0.25   # best quarter of training builds counts as good
_BAD_Q  = 0.75   # worst quarter counts as bad
_SHIFT  = 0.10   # an input must shift by at least 10 points to be reported
_TOP    = 5      # how many inputs to explain


def _load(train_csv: str) -> ezr.TBL:
  "Read TRAIN_CSV into an ezr table, keeping file order."
  return ezr.Tbl(ezr.csv(train_csv))


def _quantile(xs: list[float], p: float) -> float:
  "The p-th quantile of XS, linearly interpolated.  XS must be non-empty."
  xs     = sorted(xs)
  i      = p * (len(xs) - 1)
  lo, hi = math.floor(i), math.ceil(i)
  return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def _full(tbl: ezr.TBL, row) -> ezr.ROW:
  "Lay ROW out full-width, with '?' for anything missing; strings are cast."
  q = ["?"] * len(tbl.names)
  if isinstance(row, dict):                    # {column name: value}
    for c in tbl.cols.values():
      name = tbl.names[c.at]
      if name in row: q[c.at] = row[name]
  elif len(row) == len(tbl.names):             # already full-width
    q = list(row)
  else:                                        # positional, x-column order
    for c, v in zip(tbl.x, row): q[c.at] = v
  return [ezr.atom(v) if isinstance(v, str) else v for v in q]


def _gaps(tbl: ezr.TBL, q: ezr.ROW) -> list[tuple[ezr.COL, float]]:
  "For each goal whose value is known: (goal column, gap from its ideal)."
  out = []
  for c in tbl.y:
    v = q[c.at]
    if v != "?" and not isinstance(v, str):
      out.append((c, abs(ezr.norm(c, v) - c.goal)))
  return out


def _way(c: ezr.COL) -> str:
  "Plain words for which direction of a goal is better."
  return "higher is better" if c.goal else "lower is better"


def _shares(rs: ezr.ROWS, c: ezr.COL) -> dict:
  "Share of the known values of column C in RS, per value seen."
  vals = [r[c.at] for r in rs if r[c.at] != "?"]
  return {v: vals.count(v) / len(vals) for v in set(vals)} if vals else {}


def explain(train_csv: str) -> str:
  "Plain-words reasons, checkable on the training data, for what makes builds good."
  tbl  = _load(train_csv)
  rows = tbl.rows
  if not rows or not tbl.y or not tbl.x:
    return "There is not enough training data to explain anything."
  ys   = [ezr.ydist(tbl, r) for r in rows]
  cut  = _quantile(ys, _GOOD_Q)
  good = [r for r, y in zip(rows, ys) if y <= cut]
  best = min(range(len(rows)), key=lambda i: ys[i])   # first of any ties
  top  = rows[best]
  out  = [
    f"Training data: {len(rows)} builds. Good means closest to the ideal "
    f"on all goals together (best quarter): {len(good)} builds, "
    f"distance to the ideal at most {cut:.2f}.",
    f"Recommended build: training row {best}, distance to the ideal {ys[best]:.2f}.",
    "  Inputs: " + ", ".join(f"{c.txt}={ezr.say(top[c.at])}" for c in tbl.x),
    "  Goals: " + ", ".join(f"{c.txt}={ezr.say(top[c.at])} ({_way(c)})"
                            for c in tbl.y),
  ]

  findings = []                                       # (size, sentence)
  for c in tbl.x:
    if isinstance(c, ezr.Num):
      gv = [ezr.norm(c, r[c.at]) for r in good if r[c.at] != "?"]
      av = [ezr.norm(c, r[c.at]) for r in rows if r[c.at] != "?"]
      if not gv or not av: continue
      mg, ma = sum(gv) / len(gv), sum(av) / len(av)
      if abs(mg - ma) >= _SHIFT:
        way = "higher" if mg > ma else "lower"
        findings.append((abs(mg - ma),
                         f"{c.txt} is {way} in the best quarter (average "
                         f"position {mg:.2f} vs {ma:.2f} overall, on a 0-to-1 scale)."))
    else:
      sg, sa = _shares(good, c), _shares(rows, c)
      if not sa: continue
      v     = max(sorted(set(sa) | set(sg), key=str),
                  key=lambda k: abs(sg.get(k, 0) - sa.get(k, 0)))
      shift = sg.get(v, 0) - sa.get(v, 0)
      if abs(shift) >= _SHIFT:
        findings.append((abs(shift),
                         f"{c.txt}={v} appears in {sg.get(v, 0):.0%} of best-quarter "
                         f"builds but {sa.get(v, 0):.0%} of all builds."))
  findings.sort(key=lambda t: -t[0])                  # stable: ties keep file order

  if findings:
    out.append("Inputs that separate the best quarter from the rest, largest shift first:")
    out += [f"  {i}. {s}" for i, (_, s) in enumerate(findings[:_TOP], 1)]
  else:
    out.append(f"No input shifts by {_SHIFT:.0%} or more between the best quarter "
               "and the rest, so the recommendation has no clear input reason.")
  out.append("These are associations in the training data, not proof of cause. "
             "Check them by measuring the recommended build against a near "
             "neighbour that differs in one input.")
  return "\n".join(out)


def why(train_csv: str, row) -> str:
  "One or two sentences: is this build good or bad, and which goal decides it."
  tbl  = _load(train_csv)
  rows = tbl.rows
  if not rows or not tbl.y:
    return "Cannot rate this build: the training data has no goals."
  if isinstance(row, int) and not isinstance(row, bool):
    row = rows[row]                                   # a training row, by index
  q    = _full(tbl, row)
  gaps = _gaps(tbl, q)
  if not gaps:
    return "Cannot rate this build: its goal values are missing."
  ys   = [ezr.ydist(tbl, r) for r in rows]
  lo   = _quantile(ys, _GOOD_Q)
  hi   = _quantile(ys, _BAD_Q)
  y    = ezr.dist([g for _, g in gaps], len(gaps))
  if y <= lo:   verdict, tier = "good", "best quarter"
  elif y >= hi: verdict, tier = "bad", "worst quarter"
  else:         verdict, tier = "middling", "middle half"
  worst, _ = max(gaps, key=lambda t: t[1])
  v    = q[worst.at]
  vals = [r[worst.at] for r in rows if r[worst.at] != "?"] or [v]
  s1 = (f"Rated {verdict}: its goals put it in the {tier} of the "
        f"{len(rows)} training builds (distance to the ideal {y:.2f}; "
        f"best-quarter cutoff {lo:.2f}, worst-quarter cutoff {hi:.2f}).")
  s2 = (f"The goal furthest from its ideal is {worst.txt} at {ezr.say(v)} "
        f"({_way(worst)}; training median {ezr.say(_quantile(vals, 0.5))}).")
  return s1 + " " + s2

