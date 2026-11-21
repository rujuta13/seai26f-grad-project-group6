# step19.py
"Step 19: spot. Flag player records that look like data-entry mistakes, and players so unusual their ratings deserve a second look."
import math
import os
import ezr

ZCAP   = 5.0    # goal z-scores are capped here, then scaled to 0..1
TYPO   = 1.0    # penalty per cell that should be a finite number but is not
UNSEEN = 1.0    # weight of the fraction of repeating input symbols never seen in train

_CACHE: dict = {}  # (path, mtime) -> (table, seen symbols per repeating input column)

def _key(train_csv: str) -> tuple:
  "Identity of TRAIN_CSV: its absolute path and when it last changed."
  path = os.path.abspath(os.path.expanduser(train_csv))
  return (path, os.path.getmtime(path))

def _isnum(v) -> bool:
  "True for a usable finite number."
  return (isinstance(v, (int, float)) and not isinstance(v, bool)
          and math.isfinite(v))

def _coerce(v):
  "Strings become atoms; other values pass through."
  return ezr.atom(v) if isinstance(v, str) else v

def _full(tbl, row) -> list:
  """Lift ROW into a full-width row. Accepts a {column name: value} dict,
  a full-width list, or a list with goal columns removed (header order).
  Any cell not given is '?'. Goal suffixes are optional in dict keys."""
  names = tbl.names
  out   = ["?"] * len(names)
  if isinstance(row, dict):
    for at, n in enumerate(names):
      if n in row:
        out[at] = _coerce(row[n])
      elif n[-1:] in ("+", "-", "!") and n[:-1] in row:
        out[at] = _coerce(row[n[:-1]])
  else:
    row = list(row)
    if len(row) == len(names):
      out = [_coerce(v) for v in row]
    else:
      keep = [i for i, n in enumerate(names) if n[-1:] not in ("+", "-", "!")]
      for i, v in zip(keep, row): out[i] = _coerce(v)
  return out

def _clean(tbl, q: list) -> tuple[list, int]:
  """Replace non-numeric or non-finite cells in numeric columns (inputs and
  goals) with '?', and count them. Such cells are typos or glitches."""
  typos = 0
  for c in tbl.cols.values():
    if type(c) is not ezr.Num: continue
    v = q[c.at]
    if v == "?": continue
    if not _isnum(v):
      q[c.at] = "?"
      typos  += 1
  return q, typos

def _load(train_csv: str):
  """Train table, and the symbol values seen per repeating input column
  (identifier-like symbols, where every value is distinct, are left out).
  Cached per file version."""
  key = _key(train_csv)
  if key not in _CACHE:
    tbl  = ezr.Tbl(ezr.csv(train_csv))
    seen = {c.at: set(c.has) for c in tbl.x
            if type(c) is ezr.Sym and len(c.has) < c.n}
    _CACHE[key] = (tbl, seen)
  return _CACHE[key]

def _score(tbl, seen: dict, row) -> float:
  "Suspicion of one ROW: far from every train row, odd goals, unseen symbols, or typos."
  q, typos = _clean(tbl, _full(tbl, row))
  if tbl.x and tbl.rows:
    nn = min(ezr.xdist(tbl, q, t) for t in tbl.rows)
  else:
    nn = 0.0
  unseen = sum(1 for at, vals in seen.items()
               if q[at] != "?" and q[at] not in vals)
  unseen = unseen / max(1, len(seen))
  zs = []
  for c in tbl.y:
    v = q[c.at]
    if type(c) is ezr.Num and _isnum(v) and c.n > 1:
      z = abs(v - c.mu) / (c.sd + 1e-32)
      zs.append(min(ZCAP, z) / ZCAP)
  goal = max(zs, default=0.0)
  return float(nn + goal + UNSEEN * unseen + TYPO * typos)

def odd(train_csv, rows) -> list[float]:
  """One suspicion score per row in ROWS (goals included), in order. Higher
  means more likely a glitch, a typo, or unlike anything in TRAIN_CSV."""
  tbl, seen = _load(train_csv)
  return [_score(tbl, seen, r) for r in rows]

