# step6.py
"Step 6: justify. Plain reasons for good builds, and a one-sentence verdict per build."
import ezr

QUANTILE = 0.25   # a build is "good" if its goals are in the best 25% of train
TOP      = 5      # flag settings to report in explain
MINSUP   = 5      # a flag setting must occur in at least this many builds
LIFT     = 1.2    # ...and be this much more common among good builds

def _load(train_csv: str):
  "Table of TRAIN_CSV, with columns and goals from its header."
  return ezr.Tbl(ezr.csv(train_csv))

def _coerce(v):
  "Strings become atoms; other values pass through."
  return ezr.atom(v) if isinstance(v, str) else v

def _full(tbl, row) -> list:
  """Lift ROW into a full-width row. Accepts a {column name: value} dict,
  a full-width list, or a list with goal columns removed (header order).
  Any cell not given is '?'."""
  names = tbl.names
  out   = ["?"] * len(names)
  if isinstance(row, dict):
    for at, n in enumerate(names):
      if n in row: out[at] = _coerce(row[n])
    return out
  row = list(row)
  if len(row) == len(names):
    return [_coerce(v) for v in row]
  keep = [i for i, n in enumerate(names) if n[-1] not in "+-!"]
  for i, v in zip(keep, row): out[i] = _coerce(v)
  return out

def _ydist(tbl, row) -> float | None:
  "Distance of ROW's numeric goals from their best; None if no goal is known."
  sq, n = 0.0, 0
  for c in tbl.y:
    if type(c) is not ezr.Num: continue
    v = row[c.at]
    if v == "?": continue
    sq += (ezr.norm(c, v) - c.goal) ** 2
    n  += 1
  return (sq / n) ** .5 if n else None

def _cut(tbl):
  "Sorted known goal distances of TBL's rows, and the distance cutoff for good."
  ys  = sorted(d for d in (_ydist(tbl, r) for r in tbl.rows) if d is not None)
  thr = ys[int(QUANTILE * (len(ys) - 1))] if ys else float("inf")
  return ys, thr

def _goal_text(c) -> str:
  "Goal name without its suffix, and its direction."
  return f"{c.txt[:-1]} ({'maximise' if c.goal else 'minimise'})"

def _paths(tbl, node, conds: tuple = ()):
  "Yield (leaf, conditions) for every leaf under NODE, conditions in words."
  if not node.kids:
    yield node, conds
    return
  name    = tbl.names[node.at]
  is_sym  = type(tbl.cols[node.at]) is ezr.Sym
  yes     = f"{name} {'=' if is_sym else '<='} {ezr.say(node.v)}"
  no      = f"{name} {'!=' if is_sym else '>'} {ezr.say(node.v)}"
  yield from _paths(tbl, node.kids[0], conds + (yes,))
  yield from _paths(tbl, node.kids[1], conds + (no,))

def explain(train_csv) -> str:
  "Plain reasons for what makes builds good; each can be checked by counting."
  tbl     = _load(train_csv)
  ys, thr = _cut(tbl)
  if not ys:
    return "No training build has known goal values, so nothing can be explained."
  n    = len(tbl.rows)
  good = [d is not None and d <= thr
          for d in (_ydist(tbl, r) for r in tbl.rows)]
  base = sum(good) / n
  goals = ", ".join(_goal_text(c) for c in tbl.y)
  out  = [f"A build is good when its goals ({goals}) are in the best "
          f"{round(100 * QUANTILE)}% of the {n} training builds. Distance 0 is "
          f"ideal; good means distance at or below {ezr.say(thr)}, and the mean "
          f"over all builds is {ezr.say(sum(ys) / len(ys))}."]

  complete = [r for r in tbl.rows if all(r[c.at] != "?" for c in tbl.y)]
  conds: tuple = ()
  leaf = None
  if complete:
    ls   = list(_paths(tbl, ezr.tree(tbl, complete)))
    pool = [p for p in ls if len(p[0].rows) >= ezr.the.Leaf] or ls
    leaf, conds = min(pool, key=lambda p: p[0].mu)
  if conds:
    out.append(f"The clearest group is where {' and '.join(conds)}: "
               f"{len(leaf.rows)} of {len(complete)} builds, with mean "
               f"distance {ezr.say(leaf.mu)}.")
  else:
    out.append("No single input setting splits the builds into a clearly "
               "better group, so no rule is claimed.")

  hits = []
  for col in tbl.x:
    for v in sorted({r[col.at] for r in tbl.rows if r[col.at] != "?"}, key=str):
      seen = [g for r, g in zip(tbl.rows, good) if r[col.at] == v]
      if len(seen) < MINSUP or base == 0: continue
      rate = sum(seen) / len(seen)
      if rate / base >= LIFT:
        hits.append((rate / base, len(seen), col.txt, v, rate))
  hits.sort(key=lambda h: (-h[0], -h[1]))
  if hits:
    out.append("Input settings more common in good builds (checkable by counting):")
    for _, k, name, v, rate in hits[:TOP]:
      out.append(f"  {name} = {v}: {rate:.0%} of the {k} builds with it are "
                 f"good, versus {base:.0%} overall.")
  out.append("These are associations in the training data, not proof that "
             "a setting causes the goals.")
  return "\n".join(out)

def why(train_csv, row) -> str:
  "One sentence: is this build good or bad, and why, judged by its goals."
  tbl     = _load(train_csv)
  ys, thr = _cut(tbl)
  if not ys:
    return ("This build cannot be rated: the training file has no known "
            "goal values.")
  full  = _full(tbl, row)
  known = [(c, full[c.at]) for c in tbl.y if full[c.at] != "?"]
  if not known:
    return "This build cannot be rated: it gives no goal values."
  d = _ydist(tbl, full)
  if d is None:
    return "This build cannot be rated: its goal values are not numeric."
  verdict = "good" if d <= thr else "bad"
  beat    = sum(1 for y in ys if y > d) / len(ys)
  vals    = ", ".join(f"{c.txt[:-1]}={ezr.say(v)}" for c, v in known)
  return (f"This build is {verdict}: its goals ({vals}) sit closer to the ideal "
          f"than {beat:.0%} of the {len(ys)} training builds (distance "
          f"{ezr.say(d)}, good means at or below {ezr.say(thr)}).")

