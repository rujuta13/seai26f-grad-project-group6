# step9.py
"Step 9: odd. Flag benchmark results that look like glitches, and flag flag sets unlike anything in train."
import math
import ezr

ZCAP   = 5.0    # goal z-scores are capped here, then scaled to 0..1
TYPO   = 1.0    # penalty per cell that should be a finite number but is not
UNSEEN = 1.0    # weight of the fraction of input symbols never seen in train

_CACHE: dict = {}

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

def _load(train_csv: str):
  "Train table, and the symbol values seen per input column; cached per file."
  if train_csv not in _CACHE:
    tbl  = ezr.Tbl(ezr.csv(train_csv))
    seen = {c.at: set(c.has) for c in tbl.x if type(c) is ezr.Sym}
    _CACHE[train_csv] = (tbl, seen)
  return _CACHE[train_csv]

def _clean(tbl, q: list) -> tuple[list, int]:
  """Replace non-numeric or non-finite cells in numeric columns with '?',
  and count them. Such cells are typos or glitches."""
  typos = 0
  for c in list(tbl.x) + list(tbl.y):
    if type(c) is not ezr.Num: continue
    v = q[c.at]
    if v == "?": continue
    if not (isinstance(v, (int, float)) and math.isfinite(v)):
      q[c.at] = "?"
      typos  += 1
  return q, typos

def _score(tbl, seen: dict, row) -> float:
  "Suspicion of one ROW: far from every train row, odd goals, or typos."
  q, typos = _clean(tbl, _full(tbl, row))
  if tbl.x:
    nn = min((ezr.xdist(tbl, q, t) for t in tbl.rows), default=1.0)
  else:
    nn = 0.0
  unseen = sum(1 for c in tbl.x
               if type(c) is ezr.Sym and q[c.at] != "?"
               and q[c.at] not in seen[c.at])
  unseen = unseen / max(1, len(tbl.x))
  zs = []
  for c in tbl.y:
    v = q[c.at]
    if type(c) is ezr.Num and v != "?" and c.n > 1:
      z = abs(v - c.mu) / (c.sd + 1e-32)
      zs.append(min(ZCAP, z) / ZCAP)
  goal = max(zs, default=0.0)
  return float(nn + goal + UNSEEN * unseen + TYPO * typos)

def odd(train_csv, rows) -> list[float]:
  """One suspicion score per row in ROWS, in order. Higher means more likely
  a glitch, a typo, or unlike anything in TRAIN_CSV."""
  tbl, seen = _load(train_csv)
  return [_score(tbl, seen, r) for r in rows]

