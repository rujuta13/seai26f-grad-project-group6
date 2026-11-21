# step28.py: fix. Fewest input flags to change so a bad row becomes good, keeping frozen ones.
"step28.py: fix. Fewest input flags to change so a bad row becomes good, keeping frozen ones."
import functools
import math
import ezr

GOOD_PCT = 25   # a row is "good" if its goal distance is in the best 25% of training

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
  """ROW in TBL's column order. A dict may omit a column or key it without its goal
  suffix; missing cells are '?'."""
  n = len(tbl.names)
  if isinstance(row, dict):
    raw = []
    for name in tbl.names:
      key = str(name)
      if key in row:                                     raw.append(row[key])
      elif key and key[-1] in "+-!" and key[:-1] in row: raw.append(row[key[:-1]])
      else:                                              raw.append("?")
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

def fix(train_csv, row, frozen=()) -> dict:
  """Inputs of ROW (a bad row, goals removed), with as few values changed as possible so
  it becomes good. Each candidate is a good training flag set, whose goals are known; the
  cost of a candidate is the number of input columns where ROW differs from it. Columns
  that already agree keep ROW's value (equal by construction). FROZEN names input columns
  that must not change. Ties go to fewer changes, then to the better goal distance.
  Raises ValueError if a frozen name is not an input column, or if no good training flag
  set agrees with the frozen values."""
  tbl, known, cut = _model(train_csv)
  xs     = list(tbl.x)
  inputs = {c.txt for c in xs}
  if frozen is None: frozen = ()
  if isinstance(frozen, str): frozen = (frozen,)
  frozen = set(frozen)
  if frozen - inputs:
    raise ValueError(f"frozen names are not input columns: {sorted(frozen - inputs)}; "
                     f"inputs are {sorted(inputs)}")
  if not known:
    raise ValueError("no training row has all its goals known, so nothing is good")
  full = _align(tbl, row)
  mine = {c.txt: full[c.at] for c in xs}

  best = None
  for y, i in known:                # known is sorted by y, so stop at the first non-good
    if y > cut: break
    cand = tbl.rows[i]
    changes, legal = 0, True
    for c in xs:
      cv, mv = cand[c.at], mine[c.txt]
      if cv == mv: continue
      if c.txt in frozen or cv == "?":   # frozen values must not move; "?" is no setting
        legal = False
        break
      changes += 1
    if not legal: continue
    key = (changes, y, i)           # fewest changed inputs, then closest to ideal
    if best is None or key < best[0]: best = (key, i)
  if best is None:
    raise ValueError(f"no good training flag set agrees with the frozen values {sorted(frozen)}")
  cand = tbl.rows[best[1]]
  return {c.txt: cand[c.at] for c in xs}

