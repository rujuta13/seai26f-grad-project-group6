# step19.py
#!/usr/bin/env python3 -B
"Flag player records that look like data-entry mistakes, or are so unlike the rest that their ratings deserve a second look, via ezr."
import math
import ezr

GOAL_SCALE = 3.0                  # a value this many sds from the training mean scores 1
Z_CAP      = 10.0                 # sds beyond this count as this many, so one cell cannot dominate
TEXT_SCORE = Z_CAP / GOAL_SCALE   # a word, or nan/inf, where a number belongs

_cache: dict[str, tuple] = {}     # train_csv -> (head, kinds, cols, seen, xs, flags, enc)

def _num(v) -> bool:
  "True for an int or float, but not a bool."
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def _cell(v):
  "V as one cell: '?' if blank or unknown; otherwise its atom (a typo stays text)."
  if v is None: return "?"
  if isinstance(v, str):
    v = v.strip()
    return "?" if v in ("", "?") else ezr.atom(v)
  return v

def _kinds(head: list, rows: list) -> list:
  "How each column is read: 'goal', 'num', 'sym', or None (ignored)."
  n     = len(rows)
  kinds = []
  for at, name in enumerate(head):
    vals = [r[at] for r in rows if r[at] != "?"]
    nums = [v for v in vals if _num(v) and math.isfinite(v)]
    if name == "num" or name[-1:] == "X" or not vals:
      kind = None                                  # row id, ignored column, or empty
    elif name[-1] in "+-":
      kind = "goal" if nums else None
    elif len(nums) * 2 >= len(vals):
      kind = "num"                                 # mostly numbers: text cells are typos
    elif len({str(v) for v in vals}) > max(1, n // 2):
      kind = None                                  # an identifier: no signal
    else:
      kind = "sym"
    kinds.append(kind)
  return kinds

def _enc(kind, col, v):
  "V as used for distances: None if unknown, a symbol as is, a number scaled 0..1 by ezr.norm."
  if v == "?": return None
  if kind == "sym": return v
  if not _num(v) or not math.isfinite(v): return None
  return ezr.norm(col, v)

def _load(train_csv: str) -> tuple:
  "Read TRAIN_CSV once; return its header, column kinds, number stats, symbol sets, input columns, and scaled training rows."
  if train_csv not in _cache:
    src   = ezr.csv(train_csv)
    first = next(src, None)
    head  = ([] if first is None
             else [str(v).strip() or f"col{at}" for at, v in enumerate(first)])
    width = len(head)
    rows  = []
    for r in src:
      r = [_cell(v) for v in r]
      rows.append((r + ["?"] * width)[:width])
    kinds = _kinds(head, rows)
    cols, seen = {}, {}
    for at, kind in enumerate(kinds):
      if kind in ("goal", "num"):
        cols[at] = ezr.Num(head[at], at)
        for r in rows:
          if _num(r[at]) and math.isfinite(r[at]): ezr.add(cols[at], r[at])
      elif kind == "sym":
        seen[at] = {r[at] for r in rows if r[at] != "?"}
    xs    = [at for at, kind in enumerate(kinds) if kind in ("num", "sym")]
    flags = [kinds[at] == "sym" for at in xs]
    enc   = [[_enc(kinds[at], cols.get(at), r[at]) for at in xs] for r in rows]
    _cache[train_csv] = (head, kinds, cols, seen, xs, flags, enc)
  return _cache[train_csv]

def _align(head: list, row: dict) -> list:
  "ROW's cells aligned with HEAD; a missing or blank cell is '?'."
  return [_cell(row.get(name, "?")) for name in head]

def _unusual(kind, col, seen, v) -> float:
  "How unlike the training values of one column V is: 0 is typical, higher is stranger."
  if v == "?": return 0.0                          # unknown is not, by itself, a glitch
  if kind == "sym": return 0.0 if v in seen else 1.0
  if not _num(v) or not math.isfinite(v): return TEXT_SCORE
  if col.n == 0: return 0.0                        # nothing to compare with
  if col.sd > 0: z = abs(v - col.mu) / col.sd
  else:          z = 0.0 if v == col.mu else Z_CAP
  return min(z, Z_CAP) / GOAL_SCALE

def _joint(flags: list, enc: list, q: list) -> float:
  "Distance from the scaled inputs Q to the nearest training row ENC, over the input columns."
  if not enc: return 1.0                           # no training rows: unlike anything
  if not flags: return 0.0                         # no inputs: nothing to be unlike
  m, best = len(flags), math.inf
  for t in enc:
    total = 0.0
    for sym, a, b in zip(flags, q, t):
      if a is None or b is None: g = 1.0           # unknown = far, as in ezr
      elif sym: g = 0.0 if a == b else 1.0
      else:     g = abs(a - b)
      total += g * g
    best = min(best, math.sqrt(total / m))
    if best == 0.0: break
  return best

def odd(train_csv, rows) -> list[float]:
  """One suspicion score per row in ROWS, goals included: higher means more likely
  a glitch, a typo, or unlike anything in TRAIN_CSV. Each row's score is the largest of:
  its distance to the nearest training row over the input columns; and its most
  unusual cell in any input or goal column (a symbol never seen in training, a number
  far from the training mean, or text or nan/inf where a number belongs). Unknown
  cells ('?') never raise a score by themselves."""
  head, kinds, cols, seen, xs, flags, enc = _load(train_csv)
  out = []
  for row in rows:
    vals    = _align(head, row)
    s_cell  = max((_unusual(kinds[at], cols.get(at), seen.get(at), vals[at])
                   for at in range(len(head)) if kinds[at] is not None),
                  default=0.0)
    q       = [_enc(kinds[at], cols.get(at), vals[at]) for at in xs]
    s_joint = _joint(flags, enc, q)
    out.append(float(max(s_joint, s_cell)))
  return out

