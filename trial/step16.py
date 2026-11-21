# step16.py: justify. Plain reasons a skeptic can check, and one-sentence verdicts.
"step16.py: justify. Plain reasons a skeptic can check, and one-sentence verdicts."
import functools
import math
import ezr

GOOD_PCT = 25     # a row is "good" if its goal distance is in the best 25% of training
MIN_SIDE = 5      # fewest rows on each side of a split we will report
TOP      = 5      # how many input columns to explain

def _val(v):
  "Cells from dicts may still be strings: coerce them the way csv does."
  return ezr.atom(v) if isinstance(v, str) else v

def _align(tbl, row):
  "ROW in TBL's column order, from a dict by column name or from a list; missing cells are '?'."
  if isinstance(row, dict):
    return [_val(row.get(n, "?")) for n in tbl.names]
  return [_val(v) for v in row]

def _ydist(tbl, row):
  "How far ROW's goals are from ideal; infinite if there are no goals or any goal is unknown."
  if not tbl.y or any(row[c.at] == "?" for c in tbl.y): return math.inf
  return ezr.ydist(tbl, row)

@functools.lru_cache(maxsize=8)
def _model(train_csv):
  "The table, its rated rows as (ydist, index) sorted best first, and the cut for good."
  tbl   = ezr.Tbl(ezr.csv(train_csv))
  rated = [(_ydist(tbl, r), i) for i, r in enumerate(tbl.rows)]
  known = sorted(t for t in rated if t[0] < math.inf)
  cap   = max(1, len(known) * GOOD_PCT // 100)
  cut   = known[min(cap, len(known)) - 1][0] if known else -math.inf
  return tbl, known, cut

def _rate(ids, good):
  "Share of IDS that are good."
  return sum(i in good for i in ids) / len(ids)

def _effect(tbl, col, ids, good):
  "(strength, sentence) for the clearest split of COL among IDS, or None."
  vals   = {i: tbl.rows[i][col.at] for i in ids}
  cohort = [i for i in ids if vals[i] != "?"]
  if type(col) is ezr.Sym:
    best = None
    for v in {vals[i] for i in cohort}:
      inn  = [i for i in cohort if vals[i] == v]
      outs = [i for i in cohort if vals[i] != v]
      if len(inn) < MIN_SIDE or len(outs) < MIN_SIDE: continue
      a, b = _rate(inn, good), _rate(outs, good)
      cand = (abs(a - b),
              f"rows with {col.txt}={v} are good {a:.0%} of the time, "
              f"other rows {b:.0%} (n={len(inn)} vs {len(outs)}).")
      if best is None or cand[0] > best[0]: best = cand
    return best
  xs = sorted(vals[i] for i in cohort)
  if len(xs) < 2 * MIN_SIDE: return None
  m  = xs[len(xs) // 2]
  lo = [i for i in cohort if vals[i] <= m]
  hi = [i for i in cohort if vals[i] > m]
  if len(lo) < MIN_SIDE or len(hi) < MIN_SIDE: return None
  a, b = _rate(hi, good), _rate(lo, good)
  return (abs(a - b),
          f"rows with {col.txt} above {m:g} are good {a:.0%} of the time, "
          f"rows at or below it {b:.0%} (n={len(hi)} vs {len(lo)}).")

def explain(train_csv) -> str:
  "Reasons, in plain words a skeptical expert can check, for what makes rows good."
  tbl, known, cut = _model(train_csv)
  if not known:
    return "No training row has all its goals known, so nothing can be recommended or explained."
  rows         = tbl.rows
  good         = {i for y, i in known if y <= cut}
  ids          = [i for _, i in known]
  top_y, top_i = known[0]
  goals = ", ".join(f"{c.txt[:-1]} ({'minimise' if c.txt[-1] == '-' else 'maximise'})"
                    for c in tbl.y)
  goal_vals = ", ".join(f"{c.txt}={rows[top_i][c.at]}" for c in tbl.y)
  flags     = ", ".join(f"{c.txt}={rows[top_i][c.at]}" for c in tbl.x)
  lines = [
    f"Recommended flag set: training row {top_i}, the closest to ideal "
    f"(goal distance {ezr.say(top_y, 3)}, where 0 is ideal).",
    f"  Goals: {goals}.",
    f"  Its goal values: {goal_vals}.",
    f"  Its flags: {flags}.",
    f"Good means: goal distance in the best {GOOD_PCT}% of the {len(known)} rows "
    f"with all goals known ({len(good)} rows; cut {ezr.say(cut, 3)}).",
    "What makes rows good, strongest first (each figure is a share of rows, "
    "so you can count them yourself):",
  ]
  effects = [e for e in (_effect(tbl, c, ids, good) for c in tbl.x) if e]
  effects.sort(key=lambda e: -e[0])
  if not effects:
    lines.append("  No input column splits the rows clearly enough to report.")
  for k, (_, text) in enumerate(effects[:TOP], 1):
    lines.append(f"  {k}. {text}")
  lines.append("These are associations in the training data, not proof of cause. "
               "Check the top ones by building the flag set and measuring it.")
  return "\n".join(lines)

def why(train_csv, row) -> str:
  "One or two sentences on why this row (goals included) is good or bad."
  tbl, known, cut = _model(train_csv)
  if not known:
    return "Cannot judge: no training row has all its goals known."
  r = _align(tbl, row)
  y = _ydist(tbl, r)
  if y == math.inf:
    missing = ", ".join(c.txt for c in tbl.y if r[c.at] == "?")
    if not missing: missing = "goals"
    return f"Unrated: its goal {missing} is unknown, so it cannot be judged."
  weak    = max(tbl.y, key=lambda c: abs(ezr.norm(c, r[c.at]) - c.goal))
  verdict = "good" if y <= cut else "bad"
  place   = "within" if y <= cut else "outside"
  return (f"It is {verdict}: its goal distance from ideal is {ezr.say(y, 3)}, "
          f"{place} the best {GOOD_PCT}% of training rows (cut {ezr.say(cut, 3)}). "
          f"Its weakest goal is {weak.txt}={r[weak.at]}, "
          f"the furthest from its ideal.")

