# step6.py
#!/usr/bin/env python3 -B
"Plain reasons a skeptical expert can check, for what makes a build good, via ezr."
import ezr
import step2

def _goals(tbl) -> list:
  "The goal columns: names ending in + or -."
  return [c for c in tbl.y if c.txt[-1] in "+-"]

def _align(tbl, row: dict) -> list:
  "Turn a {name: value} dict into a list aligned with tbl.names; missing is '?'."
  out = []
  for name in tbl.names:
    v = row.get(name, "?")
    if isinstance(v, str): v = ezr.atom(v)
    out.append(v)
  return out

def _gydist(goals, vals) -> float | None:
  "Distance of VALS' goals from the ideal, or None if a goal cell is unknown."
  if any(vals[c.at] == "?" for c in goals): return None
  return ezr.dist((abs(ezr.norm(c, vals[c.at]) - c.goal) for c in goals),
                  len(goals))

def _labelled(tbl, goals) -> tuple[list, list]:
  "Rows with every goal known, and their goal distances."
  rows, ys = [], []
  for r in tbl.rows:
    y = _gydist(goals, r)
    if y is not None: rows.append(r); ys.append(y)
  return rows, ys

def explain(train_csv) -> str:
  "Single input conditions that good rows share, with the rates to check them."
  tbl   = ezr.Tbl(ezr.csv(train_csv))
  goals = _goals(tbl)
  if not goals or not tbl.x:
    return "Nothing to explain: the table needs goal columns and input columns."
  rows, ys = _labelled(tbl, goals)
  n = len(rows)
  if n == 0:
    return "Nothing to explain: no training row has every goal cell known."
  k     = max(1, n // 4)                       # good = the best quarter
  order = sorted(range(n), key=lambda i: ys[i])
  good  = set(order[:k])
  cut   = ys[order[k-1]]
  base  = len(good) / n
  leaf  = max(1, ezr.the.Leaf)
  found = []
  for col in tbl.x:
    name = col.txt
    best = None
    for v, go in ezr.candidates(col, rows):
      yes  = [i for i in range(n) if go(rows[i])]
      seen = set(yes)
      no   = [i for i in range(n) if i not in seen]
      if type(col) is ezr.Sym:
        texts = (f"{name} = {ezr.say(v)}", f"{name} != {ezr.say(v)}")
      else:
        texts = (f"{name} <= {ezr.say(v)}", f"{name} > {ezr.say(v)}")
      for side, text in zip((yes, no), texts):
        if len(side) < leaf: continue
        hits = sum(1 for i in side if i in good)
        rate = hits / len(side)
        lift = rate - base
        if best is None or lift > best[0]:
          best = (lift, text, rate, len(side))
    if best and best[0] > 0: found.append(best)
  found.sort(key=lambda p: -p[0])
  goal_names = ", ".join(g.txt for g in goals)
  lines = [
    f"Goals, where lower distance is better: {goal_names}.",
    f"Good means the best quarter of the {n} training rows: goal distance "
    f"{cut:.2f} or less, against an overall rate of {base:.0%}.",
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
  "One or two sentences: is ROW good, judged by its goals against the training cutoff."
  tbl   = ezr.Tbl(ezr.csv(train_csv))
  goals = _goals(tbl)
  if not goals:
    return "This build cannot be rated: the table has no goal columns."
  _, ys = _labelled(tbl, goals)
  if not ys:
    return "This build cannot be rated: no training row has every goal known."
  ys  = sorted(ys)
  cut = ys[max(1, len(ys) // 4) - 1]
  vals = _align(tbl, row)
  goal_names = {g.txt for g in goals}
  missing = [g for g in goals if vals[g.at] == "?"]
  estimated = []
  if missing:
    clean = {k: (ezr.atom(v) if isinstance(v, str) else v)
             for k, v in row.items() if k not in goal_names}
    est = step2.guess(train_csv, clean)
    for g in missing:
      if g.txt in est:
        vals[g.at] = est[g.txt][0]
        estimated.append(g.txt)
  if any(vals[g.at] == "?" for g in goals):
    return "This build cannot be rated: some of its goal values are unknown."
  d = _gydist(goals, vals)
  good = d <= cut
  s1 = (f"This build is {'good' if good else 'bad'}: its goals sit {d:.2f} "
        f"from the ideal, {'inside' if good else 'outside'} the best-quarter "
        f"cutoff of {cut:.2f} over {len(ys)} training rows.")
  if not estimated: return s1
  return (s1 + f" {', '.join(estimated)} was not given, so it is estimated "
          f"as the median of its nearest labelled rows, and the rating is only "
          f"as good as that estimate.")

