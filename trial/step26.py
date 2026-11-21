# step26.py: justify. Plain reasons a skeptic can check, and one-sentence verdicts.
"step26.py: justify. Plain reasons a skeptic can check, and one-sentence verdicts."
import functools
import math
import ezr

GOOD_PCT = 25     # a row is "good" if its goal distance is in the best 25% of training
MIN_SIDE = 5      # fewest rows on each side of a split we will report
TOP      = 5      # how many input columns to explain

def _is_num(v) -> bool:
  "True if V is a usable finite number (bools and infinities do not count)."
  return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def _val(v):
  "One raw cell as an atom: blanks, None and NaN become '?'; strings are coerced like csv does."
  if v is None: return "?"
  if isinstance(v, str):
    s = v.strip()
    return "?" if s == "" else ezr.atom(s)
  if isinstance(v, float) and math.isnan(v): return "?"
  return v

def _align(tbl, row):
  "ROW in TBL's column order; a dict may omit a column or key it without its goal suffix ('?' if absent)."
  n = len(tbl.names)
  if isinstance(row, dict):
    raw = []
    for name in tbl.names:
      key = str(name)
      if key in row:                                    raw.append(row[key])
      elif key and key[-1] in "+-!" and key[:-1] in row: raw.append(row[key[:-1]])
      else:                                             raw.append("?")
  else:
    raw  = list(row)[:n]
    raw += ["?"] * (n - len(raw))
  return [_val(v) for v in raw]

def _ydist(tbl, row):
  "How far ROW's goals are from ideal; infinite if there are no goals or any goal is not a known number."
  if not tbl.y or not all(_is_num(row[c.at]) for c in tbl.y): return math.inf
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
  vals = {i: tbl.rows[i][col.at] for i in ids}
  if type(col) is ezr.Sym:
    cohort = [i for i in ids if vals[i] != "?"]
    best   = None
    for v in sorted({vals[i] for i in cohort}, key=str):   # sorted: deterministic ties
      inn  = [i for i in cohort if vals[i] == v]
      outs = [i for i in cohort if vals[i] != v]
      if len(inn) < MIN_SIDE or len(outs) < MIN_SIDE: continue
      a, b = _rate(inn, good), _rate(outs, good)
      cand = (abs(a - b),
              f"rows with {col.txt}={v} are good {a:.0%} of the time, "
              f"other rows {b:.0%} (n={len(inn)} vs {len(outs)}).")
      if best is None or cand[0] > best[0]: best = cand
    return best
  cohort = [i for i in ids if _is_num(vals[i])]
  xs = sorted(vals[i] for i in cohort)
  if len(xs) < 2 * MIN_SIDE: return None
  m  = xs[len(xs) // 2]
  lo = [i for i in cohort if vals[i] <= m]
  hi = [i for i in cohort if vals[i] > m]
  if len(lo) < MIN_SIDE or len(hi) < MIN_SIDE: return None
  a, b = _rate(hi, good), _rate(lo, good)
  return (abs(a - b),
          f"rows with {col.txt} above {ezr.say(m, 3)} are good {a:.0%} of the time, "
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
    missing = ", ".join(c.txt for c in tbl.y if not _is_num(r[c.at])) or "goals"
    return f"Unrated: its goal value(s) {missing} are missing or not numeric, so it cannot be judged."
  weak     = max(tbl.y, key=lambda c: abs(ezr.norm(c, r[c.at]) - c.goal))
  verdict  = "good" if y <= cut else "bad"
  place    = "within" if y <= cut else "outside"
  good_ids = [i for y2, i in known if y2 <= cut]
  gv       = sorted(tbl.rows[i][weak.at] for i in good_ids
                    if _is_num(tbl.rows[i][weak.at]))
  hint     = (f"; good rows typically have {weak.txt} about "
              f"{ezr.say(gv[len(gv) // 2], 3)}") if gv else ""
  return (f"It is {verdict}: its goal distance from ideal is {ezr.say(y, 3)}, "
          f"{place} the best {GOOD_PCT}% of training rows (cut {ezr.say(cut, 3)}). "
          f"Its weakest goal is {weak.txt}={r[weak.at]}, "
          f"the furthest from its ideal{hint}.")

