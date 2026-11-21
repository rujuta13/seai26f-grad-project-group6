# step6.py
import ezr

GOOD = 0.25   # a row is good if its ydist is in the best quarter of training rows

def _load(train_csv: str) -> ezr.TBL:
  "Load TRAIN_CSV as a table."
  return ezr.Tbl(ezr.csv(train_csv))

def _goals(tbl: ezr.TBL) -> list:
  "The numeric goal columns of TBL (names ending + or -)."
  return [c for c in tbl.y if type(c) is ezr.Num]

def _complete(goals: list, q: ezr.ROW) -> bool:
  "True if every goal cell of Q is a number."
  return all(isinstance(q[c.at], (int, float)) for c in goals)

def _ydist(goals: list, q: ezr.ROW) -> float:
  "How far Q's goals are from the best they could be (0 is best, 1 is worst)."
  return ezr.dist((abs(ezr.norm(c, q[c.at]) - c.goal) for c in goals),
                  len(goals))

def _align(tbl: ezr.TBL, row) -> ezr.ROW:
  """Align ROW to TBL's columns. ROW is a dict by column name, or a list
  either over every column or over the non-goal columns. Missing is '?'."""
  q = ["?"] * len(tbl.names)
  if isinstance(row, dict):
    pairs = [(at, row.get(name)) for at, name in enumerate(tbl.names)]
  elif len(row) == len(tbl.names):
    pairs = list(enumerate(row))
  else:
    keep  = [at for at, n in enumerate(tbl.names) if n[-1] not in "+-"]
    pairs = list(zip(keep, row))
  for at, v in pairs:
    if v is None or (isinstance(v, str) and v.strip() in ("", "?")): continue
    q[at] = ezr.atom(v) if isinstance(v, str) else v
  return q

def _rate(tbl: ezr.TBL, goals: list):
  """Rows with complete goals, whether each is good, the ydist cutoff for
  good, and the share of good rows."""
  rows = [r for r in tbl.rows if _complete(goals, r)]
  if not rows: return [], [], 0.0, 0.0
  ys   = [_ydist(goals, r) for r in rows]
  cut  = sorted(ys)[min(len(ys) - 1, int(GOOD * (len(ys) - 1)))]
  good = [y <= cut for y in ys]
  base = sum(good) / len(good)
  return rows, good, cut, base

def _label(col: ezr.COL, name: str, v) -> str:
  "A readable name for one split of COL."
  if type(col) is ezr.Sym: return f"{name} = {ezr.say(v)}"
  return f"{name} <= {ezr.say(v)}"

def _settings(tbl: ezr.TBL, rows: ezr.ROWS, good: list) -> list:
  """Every input split with at least the.Leaf rows, as
  (name, label, go, n, share_good)."""
  out = []
  for col in tbl.x:
    name = tbl.names[col.at]
    for v, go in ezr.candidates(col, rows):
      hits = [i for i, r in enumerate(rows) if go(r)]
      n    = len(hits)
      if n < ezr.the.Leaf: continue
      share = sum(good[i] for i in hits) / n
      out.append((name, _label(col, name, v), go, n, share))
  return out

def _recommend(settings: list, base: float) -> list:
  "For each input, its best setting, if it beats the baseline share of good."
  best: dict[str, tuple] = {}
  for s in settings:
    if s[4] > base and (s[0] not in best or s[4] > best[s[0]][4]):
      best[s[0]] = s
  return sorted(best.values(), key=lambda s: -s[4])

def _flag_note(rec: list, q: ezr.ROW) -> str:
  "One sentence: how many recommended settings ROW's flags match."
  if not rec:
    return ("No input setting separates good rows here, "
            "so the flags give no reason either way.")
  met    = [s[1] for s in rec if s[2](q)]
  missed = [s[1] for s in rec if not s[2](q)]
  note   = f"Its flags match {len(met)} of {len(rec)} recommended settings"
  if missed: note += f"; it lacks {', '.join(missed[:2])}"
  return note + "."

def explain(train_csv: str) -> str:
  "Plain-words reasons, checkable by hand, for what makes rows good."
  tbl   = _load(train_csv)
  goals = _goals(tbl)
  if not goals or not tbl.x:
    return ("Cannot explain: the file needs goal columns (ending + or -) "
            "and input columns.")
  rows, good, cut, base = _rate(tbl, goals)
  if not rows:
    return "Cannot explain: no training row has numeric values for every goal."
  gnames   = ", ".join(tbl.names[c.at] for c in goals)
  settings = _settings(tbl, rows, good)
  rec      = _recommend(settings, base)
  lines = [
    f"1. What 'good' means: a row is good when its goals ({gnames}) are in "
    f"the best {GOOD:.0%} of the {len(rows)} training rows. Distance (ydist) "
    f"is 0 at the best possible goal values and 1 at the worst, so good means "
    f"ydist <= {cut:.2f}. Here {base:.0%} of rows are good.",
    "2. Recommended flag set. For each input, the setting whose rows are good "
    "most often, if it beats the baseline and has enough rows:",
  ]
  if rec:
    for _, label, _, n, share in rec:
      lines.append(f"   - {label}: {n} rows, {share:.0%} good "
                   f"(baseline {base:.0%}).")
    hits = [i for i, r in enumerate(rows) if all(s[2](r) for s in rec)]
    if hits:
      hs = sum(good[i] for i in hits) / len(hits)
      lines.append(f"   Rows with all of these together: {len(hits)}, "
                   f"{hs:.0%} good.")
    else:
      lines.append("   No training row has all of these together, "
                   "so the combined set is untested.")
  else:
    lines.append("   No single setting beats the baseline, "
                 "so no flag set is recommended from this data.")
  worst = sorted((s for s in settings if s[4] < base), key=lambda s: s[4])[:3]
  lines.append("3. Settings linked to bad rows (lowest good share):")
  for _, label, _, n, share in worst:
    lines.append(f"   - {label}: {n} rows, {share:.0%} good.")
  if not worst: lines.append("   None.")
  lines.append("4. Limits: these are patterns in one training file, not proof "
               "that a flag causes better goals. Confirm the recommended set by "
               "building a few rows with and without each flag and measuring "
               "the goals.")
  return "\n".join(lines)

def why(train_csv: str, row) -> str:
  "One or two sentences: why ROW is good or bad, and which flags back that up."
  tbl   = _load(train_csv)
  goals = _goals(tbl)
  if not goals or not tbl.x:
    return "Cannot rate this row: the file needs goal columns and input columns."
  rows, good, cut, base = _rate(tbl, goals)
  if not rows:
    return "Cannot rate this row: no training row has numeric goals."
  q   = _align(tbl, row)
  rec = _recommend(_settings(tbl, rows, good), base)
  if not _complete(goals, q):
    missing = [tbl.names[c.at] for c in goals
               if not isinstance(q[c.at], (int, float))]
    first = (f"Cannot rate it as good or bad: its goal values are missing "
             f"({', '.join(missing)}).")
  else:
    y = _ydist(goals, q)
    if y <= cut:
      first = (f"Good: its goals are within the best {GOOD:.0%} of training "
               f"rows (ydist {y:.2f} <= {cut:.2f}, where 0 is best).")
    else:
      first = (f"Bad: its goals fall outside the best {GOOD:.0%} of training "
               f"rows (ydist {y:.2f} > {cut:.2f}, where 0 is best).")
  return f"{first} {_flag_note(rec, q)}"

