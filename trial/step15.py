# step15.py
#!/usr/bin/env python3 -B
"Input columns to blame for bad rows (bad on the goals), via ezr."
import math
import ezr

def blame(train_csv) -> list[str]:
  """Input column names, most to blame for bad rows (bad on the goals) first.
  A row is bad if its goals are in the worst quarter by goal distance (ydist).
  Text columns count as inputs unless they are identifiers (too many distinct
  values) or goals; the row id column 'num' is ignored. Each input column is
  scored by its largest lift: how much more often bad rows fall on one side of
  a candidate split than they fall in the table overall. Rows with an unknown
  goal cell are ignored. Ties keep file order."""

  def cell(v):
    "One cell: '?' if blank, unknown, or not finite; otherwise its value."
    if isinstance(v, str): v = v.strip()
    if v in ("", "?"): return "?"
    if isinstance(v, float) and not math.isfinite(v): return "?"
    return v

  src   = ezr.csv(train_csv)
  first = next(src, None)
  if first is None: return []
  head  = [str(v).strip() or f"col{at}" for at, v in enumerate(first)]
  width = len(head)
  rows  = []
  for r in src:
    r = [cell(v) for v in r]
    rows.append((r + ["?"] * width)[:width])
  n = len(rows)

  # classify each column: goal, numeric input, text input, or ignored
  kinds = []
  for at, name in enumerate(head):
    vals = [r[at] for r in rows if r[at] != "?"]
    goal = name[-1] in "+-"
    if name == "num" or not vals:
      kind = None
    elif any(isinstance(v, (str, bool)) for v in vals):
      distinct = len({str(v) for v in vals})
      kind = None if goal or distinct > max(1, n // 2) else "sym"
    else:
      kind = "goal" if goal else "num"
    kinds.append(kind)

  # ezr names chosen so ezr reads each kind correctly; original names kept in head
  names = []
  for at, kind in enumerate(kinds):
    names.append({"goal": f"G{at}{head[at][-1]}",
                  "num":  f"N{at}",
                  "sym":  f"s{at}"}.get(kind, f"Z{at}X"))
  data = [[(str(v) if v != "?" else v) if kind == "sym" else v
           for v, kind in zip(r, kinds)] for r in rows]

  tbl   = ezr.Tbl([names] + data)
  goals = [c for c in tbl.y if c.txt[-1] in "+-"]
  xs    = tbl.x
  if not goals or not xs: return []

  def gydist(row) -> float | None:
    "Distance of ROW's goals from the ideal, or None if a goal cell is unknown."
    if any(row[c.at] == "?" for c in goals): return None
    return ezr.dist((abs(ezr.norm(c, row[c.at]) - c.goal) for c in goals),
                    len(goals))

  kept, ys = [], []
  for r in data:
    y = gydist(r)
    if y is not None:
      kept.append(r)
      ys.append(y)
  m = len(kept)
  if m == 0: return [head[c.at] for c in xs]

  k     = max(1, m // 4)                         # the worst quarter
  order = sorted(range(m), key=lambda i: ys[i])
  bad   = set(order[-k:])                        # highest ydist = worst
  base  = len(bad) / m                           # bad rate overall
  leaf  = max(1, ezr.the.Leaf)                   # min rows on a side

  def score(col) -> float:
    "Largest lift in bad rate of any side of any candidate split of COL."
    best = -1.0                                  # lift is always >= -1
    for _, go in ezr.candidates(col, kept):
      yes  = [i for i in range(m) if go(kept[i])]
      seen = set(yes)
      no   = [i for i in range(m) if i not in seen]
      for side in (yes, no):
        if len(side) >= leaf:
          rate = sum(i in bad for i in side) / len(side)
          best = max(best, rate - base)
    return best

  scored = [(score(c), c) for c in xs]
  scored.sort(key=lambda p: -p[0])               # stable: ties keep file order
  return [head[c.at] for _, c in scored]

