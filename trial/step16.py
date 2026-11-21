# step16.py
#!/usr/bin/env python3 -B
"Plain reasons a skeptical expert can check, for what makes a build good, via ezr."
import math
import ezr
import step12

SKIP = {"num", "Rank"}   # row id, and a rank derived from OVR: not reasons

def _cell(v):
  "V as one cell: '?' if blank, unknown, None, or not finite; otherwise its atom."
  if v is None: return "?"
  if isinstance(v, str): v = ezr.atom(v.strip() or "?")
  if isinstance(v, float) and not math.isfinite(v): return "?"
  return v

def _num(v) -> bool:
  "True for a usable number: an int or float, but not a bool."
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def _table(train_csv: str) -> tuple:
  """TRAIN_CSV as (ezr table, header names, column kinds). Goals are numeric
  columns ending in + or -; numeric inputs are 'num'; text inputs with few
  distinct values are 'sym'; identifiers, SKIP columns and empty columns are
  ignored (kind None). Text in a numeric column becomes unknown. The table
  is None if the file is empty."""
  src   = ezr.csv(train_csv)
  first = next(src, None)
  if first is None: return None, [], []
  head  = [str(v).strip() or f"col{at}" for at, v in enumerate(first)]
  width = len(head)
  rows  = []
  for r in src:
    r = [_cell(v) for v in r]
    rows.append((r + ["?"] * width)[:width])
  n = len(rows)

  kinds = []
  for at, name in enumerate(head):
    vals = [r[at] for r in rows if r[at] != "?"]
    if name in SKIP or not vals:
      kinds.append(None)
    elif name[-1] in "+-":
      kinds.append("goal" if any(_num(v) for v in vals) else None)
    elif all(_num(v) for v in vals):
      kinds.append("num")
    elif len({str(v) for v in vals}) > max(1, n // 2):
      kinds.append(None)                       # an identifier: no signal
    else:
      kinds.append("sym")

  # ezr names chosen so ezr reads each kind correctly; original names kept in head
  names = [{"goal": f"G{at}{head[at][-1]}",
            "num":  f"N{at}",
            "sym":  f"s{at}"}.get(kind, f"Z{at}X")
           for at, kind in enumerate(kinds)]
  data = []
  for r in rows:
    out = []
    for at, kind in enumerate(kinds):
      v = r[at]
      if kind == "sym":
        v = "?" if v == "?" else str(v)
      elif kind in ("num", "goal") and not _num(v):
        v = "?"
      out.append(v)
    data.append(out)
  return ezr.Tbl([names] + data), head, kinds

def _goals(tbl) -> list:
  "The goal columns: names ending in + or -."
  return [c for c in tbl.y if c.txt[-1] in "+-"]

def _align(head: list, kinds: list, row: dict) -> list:
  "Turn a {name: value} dict into a list aligned with the table; unusable cells are '?'."
  out = []
  for at, name in enumerate(head):
    kind = kinds[at]
    v    = _cell(row.get(name, "?"))
    if kind is None: v = "?"                   # ignored column: nothing to read
    elif kind == "sym": v = "?" if v == "?" else str(v)
    elif not _num(v): v = "?"
    out.append(v)
  return out

def _gydist(goals, vals):
  "Distance of VALS' goals from the ideal, or None if a goal cell is unknown."
  if any(vals[c.at] == "?" for c in goals): return None
  return ezr.dist((abs(ezr.norm(c, vals[c.at]) - c.goal) for c in goals),
                  len(goals))

def _labelled(tbl, goals) -> tuple[list, list]:
  "Rows with every goal known, and their goal distances."
  rows, ys = [], []
  for r in tbl.rows:
    y = _gydist(goals, r)
    if y is not None:
      rows.append(r)
      ys.append(y)
  return rows, ys

def _sides(col, rows, v, go) -> tuple[list, list]:
  "Row indices on the yes side of a split of COL at V, and on the no side; unknown cells are on neither."
  at  = col.at
  yes = [i for i, r in enumerate(rows) if go(r)]
  if type(col) is ezr.Sym:
    no = [i for i, r in enumerate(rows) if r[at] != "?" and r[at] != v]
  else:
    no = [i for i, r in enumerate(rows) if r[at] != "?" and r[at] > v]
  return yes, no

def explain(train_csv) -> str:
  """Single input conditions that good rows share, with the rates to check them.
  Good means the best quarter of training rows by goal distance. A condition is
  only reported if it covers at least 5% of rows and beats the overall rate."""
  tbl, head, kinds = _table(train_csv)
  if tbl is None:
    return "Nothing to explain: the training file is empty."
  goals = _goals(tbl)
  if not goals or not tbl.x:
    return "Nothing to explain: the table needs goal columns and input columns."
  rows, ys = _labelled(tbl, goals)
  n = len(rows)
  if n == 0:
    return "Nothing to explain: no training row has every goal cell known."
  k     = max(1, n // 4)                       # good = the best quarter
  order = sorted(range(n), key=lambda i: ys[i])
  cut   = ys[order[k - 1]]
  good  = {i for i in range(n) if ys[i] <= cut}
  base  = len(good) / n
  need  = max(max(1, ezr.the.Leaf), n // 20)   # enough rows to be worth a look
  found = []
  for col in tbl.x:
    name = head[col.at]
    best = None
    for v, go in ezr.candidates(col, rows):
      yes, no = _sides(col, rows, v, go)
      if type(col) is ezr.Sym:
        texts = (f"{name} = {ezr.say(v)}", f"{name} != {ezr.say(v)}")
      else:
        texts = (f"{name} <= {ezr.say(v)}", f"{name} > {ezr.say(v)}")
      for side, text in zip((yes, no), texts):
        if len(side) < need: continue
        hits = sum(1 for i in side if i in good)
        rate = hits / len(side)
        lift = rate - base
        if best is None or lift > best[0]:
          best = (lift, text, rate, len(side))
    if best and best[0] > 0: found.append(best)
  found.sort(key=lambda p: -p[0])

  goal_names = ", ".join(head[g.at] for g in goals)
  lines = [
    f"Goals, scored by distance from the ideal (lower is better): {goal_names}.",
    f"Good means the best quarter of the {n} training rows with every goal known: "
    f"goal distance {cut:.2f} or less, against an overall rate of {base:.0%}.",
  ]
  if not found:
    lines.append("No single input condition makes good rows more likely "
                 "than the overall rate, so no single flag explains them.")
  else:
    lines.append("Each line is one input condition. Good rows meet it at the "
                 "rate shown, against the overall rate. Check any line by "
                 "filtering the training CSV yourself:")
    for _, text, rate, cnt in found[:5]:
      lines.append(f"  {text}: good in {rate:.0%} of {cnt} rows "
                   f"(overall {base:.0%}).")
    lines.append("These are associations in the training data, not proof of cause.")
  return "\n".join(lines)

def why(train_csv, row) -> str:
  "One or two sentences: is ROW good, judged by its goals (including estimated ones) against the training cutoff."
  tbl, head, kinds = _table(train_csv)
  if tbl is None:
    return "This build cannot be rated: the training file is empty."
  goals = _goals(tbl)
  if not goals:
    return "This build cannot be rated: the table has no goal columns."
  _, ys = _labelled(tbl, goals)
  if not ys:
    return "This build cannot be rated: no training row has every goal known."
  ys  = sorted(ys)
  cut = ys[max(1, len(ys) // 4) - 1]
  vals = _align(head, kinds, row)
  missing = [c for c in goals if vals[c.at] == "?"]
  estimated = []
  if missing:
    goal_names = {head[c.at] for c in goals}
    clean = {k: v for k, v in row.items() if k not in goal_names}
    est = step12.guess(train_csv, clean)
    for c in missing:
      name = head[c.at]
      if name in est:
        e, lo, hi = est[name]
        vals[c.at] = float(e)
        estimated.append((name, e, lo, hi))
  if any(vals[c.at] == "?" for c in goals):
    return "This build cannot be rated: some of its goal values are unknown."
  d    = _gydist(goals, vals)
  good = d <= cut
  s1 = (f"This build is {'good' if good else 'bad'}: its goals sit {d:.2f} "
        f"from the ideal, {'inside' if good else 'outside'} the best-quarter "
        f"cutoff of {cut:.2f} over {len(ys)} training rows.")
  if not estimated: return s1
  notes = "; ".join(
    f"{name} was not given, so it is estimated at {e:.2f} (most similar builds "
    f"fall between {lo:.2f} and {hi:.2f})"
    for name, e, lo, hi in estimated)
  return f"{s1} {notes}, so the rating is only as good as that estimate."

