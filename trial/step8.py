# step8.py: fix. Fewest input flags to flip so a bad flag set becomes good, keeping frozen ones.
"step8.py: fix. Fewest input flags to flip so a bad flag set becomes good, keeping frozen ones."
import functools
import math
import ezr

GOOD_PCT = 25   # a flag set is "good" if its goal distance is in the best 25% of training

def _val(v):
  "Cells from dicts may still be strings: coerce them the way csv does."
  return ezr.atom(v) if isinstance(v, str) else v

def _ydist(tbl, row):
  "How far ROW's goals are from ideal; infinite if any goal is unknown."
  if any(row[c.at] == "?" for c in tbl.y): return math.inf
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
  """Inputs of ROW, with as few values changed as possible so it matches a good training
  flag set. ROW is a dict by column name (its goals may be absent). FROZEN names input
  columns that must not change. The result is the nearest good training flag set that
  agrees with ROW on every frozen column; ties go to the better goal distance.
  Raises ValueError if a frozen name is not an input column, or if no training flag set
  agrees with the frozen values."""
  tbl, known, cut = _model(train_csv)
  xs     = list(tbl.x)
  inputs = {c.txt for c in xs}
  if isinstance(frozen, str): frozen = (frozen,)
  frozen = set(frozen)
  if frozen - inputs:
    raise ValueError(f"frozen names are not input columns: {sorted(frozen - inputs)}; "
                     f"inputs are {sorted(inputs)}")
  if not known:
    raise ValueError("no training row has all its goals known, so nothing is good")
  full = ([_val(row.get(n, "?")) for n in tbl.names] if isinstance(row, dict)
          else [_val(v) for v in row])
  mine = {c.txt: full[c.at] for c in xs}
  good = {i for y, i in known if y <= cut}

  best = None
  for y, i in known:
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
    key = (i not in good, changes, y, i)   # good first, then fewest flips, then closest to ideal
    if best is None or key < best[0]: best = (key, i)
  if best is None:
    raise ValueError(f"no training flag set agrees with the frozen values {sorted(frozen)}")
  cand = tbl.rows[best[1]]
  return {c.txt: cand[c.at] for c in xs}

