# step9.py
#!/usr/bin/env python3 -B
"Flag benchmark rows that look like glitches, typos, or are unlike anything in the training data, via ezr."
import math
import ezr

_cache: dict[str, tuple] = {}   # train_csv -> (tbl, goal columns, input columns)

GOAL_SCALE = 3.0    # a value this many sds from the training mean scores 1
Z_CAP      = 10.0   # sds beyond this count as this many, so one cell cannot dominate

def _load(train_csv: str) -> tuple:
  "Read TRAIN_CSV once; return its table, its goal columns, and its input columns."
  if train_csv not in _cache:
    tbl   = ezr.Tbl(ezr.csv(train_csv))
    goals = [c for c in tbl.y if c.txt[-1] in "+-"]
    _cache[train_csv] = (tbl, goals, list(tbl.x))
  return _cache[train_csv]

def _cell(v):
  "V as an atom: '?' if unknown, otherwise a number or the text itself (which may be a typo)."
  if v is None: return "?"
  if isinstance(v, str): v = ezr.atom(v.strip() or "?")
  return v

def _align(tbl, row: dict) -> list:
  "Turn a {name: value} dict into a list aligned with tbl.names; missing is '?'."
  return [_cell(row.get(name, "?")) for name in tbl.names]

def _is_num(v) -> bool:
  "True for ints and floats, but not bools."
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def _unusual(col, v) -> float:
  "How unlike the training values of COL the value V is: 0 is typical, higher is stranger."
  if v == "?": return 0.0                          # unknown is not, by itself, a glitch
  if type(col) is ezr.Sym:
    return 0.0 if v in col.has else 1.0            # a symbol never seen in training
  if not _is_num(v) or not math.isfinite(v):
    return Z_CAP / GOAL_SCALE                      # a word or nan/inf where a number belongs
  if col.n == 0: return 0.0                        # nothing to compare with
  if col.sd > 0: z = abs(v - col.mu) / col.sd
  else:          z = 0.0 if v == col.mu else Z_CAP
  return min(z, Z_CAP) / GOAL_SCALE

def _gap(col, v, w) -> float:
  "Distance between one row's value V and a training value W in COL, roughly 0 to 1."
  if v == "?" or w == "?": return 1.0             # unknown = far, as in ezr
  if type(col) is ezr.Sym: return 0.0 if v == w else 1.0
  if not _is_num(v) or not math.isfinite(v): return 1.0
  return abs(ezr.norm(col, v) - ezr.norm(col, w))

def _joint(tbl, xs, vals) -> float:
  "Distance from VALS to its nearest training row, over the input columns."
  if not xs: return 0.0                           # no inputs: nothing to be unlike
  if not tbl.rows: return 1.0                     # no training rows: unlike anything
  return min(ezr.dist((_gap(c, vals[c.at], r[c.at]) for c in xs), len(xs))
             for r in tbl.rows)

def odd(train_csv, rows) -> list[float]:
  """One suspicion score per row in ROWS, goals included: higher means more likely
  a glitch, a typo, or unlike anything in TRAIN_CSV. Each row's score is the largest of:
  its joint distance to the nearest training row over the inputs; its most unusual
  input cell; and its most unusual goal cell (goals far from the training mean,
  non-numeric, or non-finite). Unknown cells ('?') never raise a score by themselves."""
  tbl, goals, xs = _load(train_csv)
  out = []
  for row in rows:
    vals = _align(tbl, row)
    s_joint = _joint(tbl, xs, vals)
    s_input = max((_unusual(c, vals[c.at]) for c in xs), default=0.0)
    s_goal  = max((_unusual(g, vals[g.at]) for g in goals), default=0.0)
    out.append(float(max(s_joint, s_input, s_goal)))
  return out

