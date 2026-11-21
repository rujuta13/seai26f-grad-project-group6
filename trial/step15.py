# step15.py
"""Step 15: blame.  Which input columns go with the bad builds?

The client asks why some players are weak from the spot (Penalties+) and
what separates the physically strong players (Strength+) from the rest.
blame() ranks the input columns over all goals together, as specified: a
row is bad if its goals together are in the worst quarter, and good if in
the best quarter.  Identifier and text columns are left out, as in step11.

Uses only the standard library plus ezr.
"""
import math
from collections import Counter
import ezr

_GOOD_Q = 0.25   # best quarter of rows counts as good
_BAD_Q  = 0.75   # worst quarter counts as bad
_DROP   = {"num", "Rank", "Name", "Nation", "League", "Team", "url"}


def _load(train_csv: str) -> ezr.TBL:
  """Read TRAIN_CSV into an ezr table.  Identifier and text columns are
  dropped; a numeric column keeps only its numbers, and anything else in
  it counts as unknown ('?')."""
  raw = list(ezr.csv(train_csv))
  if not raw: return ezr.Tbl([[]])
  head = [str(h).strip() for h in raw[0]]
  keep = [i for i, h in enumerate(head)
          if h not in _DROP and not h.endswith("X")]
  rows = []
  for r in raw[1:]:
    r   = list(r) + ["?"] * (len(head) - len(r))   # pad ragged rows
    row = []
    for i in keep:
      v = r[i]
      if head[i][:1].isupper() and not isinstance(v, (int, float)):
        v = "?"                                    # unreadable numeric cell
      row.append(v)
    rows.append(row)
  return ezr.Tbl([[head[i] for i in keep]] + rows)


def _quantile(xs: list[float], p: float) -> float:
  "The p-th quantile of XS, linearly interpolated.  XS must be non-empty."
  xs     = sorted(xs)
  i      = p * (len(xs) - 1)
  lo, hi = math.floor(i), math.ceil(i)
  return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def _ydist(goals: list, row: ezr.ROW) -> float:
  "How far ROW's goals are from their ideal: 0 is best; an unknown goal is worst."
  gaps = [1.0 if row[c.at] == "?" else abs(ezr.norm(c, row[c.at]) - c.goal)
          for c in goals]
  return math.sqrt(sum(g * g for g in gaps) / len(gaps)) if gaps else 0.0


def _shares(rs: ezr.ROWS, c: ezr.COL) -> dict:
  "Share of the known values of column C in RS, per value seen."
  vals = [r[c.at] for r in rs if r[c.at] != "?"]
  if not vals: return {}
  n = len(vals)
  return {v: k / n for v, k in Counter(vals).items()}


def blame(train_csv: str) -> list[str]:
  """Input column names, most to blame for bad rows (bad on the goals) first.
  A row is bad if its goals together are in the worst quarter, and good if
  in the best quarter.  A numeric input scores the gap between its mean
  (scaled to 0..1) in bad and good rows.  A symbolic input scores the
  largest gap, over its values, between how often that value appears in
  bad rows and in good rows.  Ties keep file order."""
  tbl   = _load(train_csv)
  rows  = tbl.rows
  goals = [c for c in tbl.y if isinstance(c, ezr.Num)]
  if not rows or not tbl.x or not goals:
    return [c.txt for c in tbl.x]

  ys   = [_ydist(goals, r) for r in rows]
  hi   = _quantile(ys, _BAD_Q)
  lo   = _quantile(ys, _GOOD_Q)
  bad  = [r for r, y in zip(rows, ys) if y >= hi]
  good = [r for r, y in zip(rows, ys) if y <= lo]
  if not bad or not good: return [c.txt for c in tbl.x]

  def score(c) -> float:
    "How strongly input column C separates bad rows from good rows."
    if isinstance(c, ezr.Num):
      vb = [ezr.norm(c, r[c.at]) for r in bad  if r[c.at] != "?"]
      vg = [ezr.norm(c, r[c.at]) for r in good if r[c.at] != "?"]
      if not vb or not vg: return 0.0
      return abs(sum(vb) / len(vb) - sum(vg) / len(vg))
    sb, sg = _shares(bad, c), _shares(good, c)
    values = set(sb) | set(sg)
    if not values: return 0.0
    return max(abs(sb.get(v, 0.0) - sg.get(v, 0.0)) for v in values)

  scored = [(score(c), i, c.txt) for i, c in enumerate(tbl.x)]
  scored.sort(key=lambda t: (-t[0], t[1]))   # most blame first; ties by file order
  return [name for _, _, name in scored]

