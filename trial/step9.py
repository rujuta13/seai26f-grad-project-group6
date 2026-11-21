# step9.py
import ezr

BAD_Z = 2.0   # a number within this many sds of its column's mean looks normal
SPAN  = 4.0   # each further sd adds to suspicion, up to a cap of 1

def _is_num(v) -> bool:
  "True if V is a real number (a bool is not)."
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def _align(tbl: ezr.TBL, row) -> ezr.ROW:
  """Align ROW to TBL's columns. ROW is a dict by column name, or a list
  either over every column or over the non-goal columns. Blank is '?';
  strings are read as atoms, so '12' becomes 12 and 'abc' stays text."""
  q = ["?"] * len(tbl.names)
  if isinstance(row, dict):
    pairs = [(at, row.get(name)) for at, name in enumerate(tbl.names)]
  elif len(row) == len(tbl.names):
    pairs = list(enumerate(row))
  else:
    keep  = [at for at, n in enumerate(tbl.names) if n[-1] not in "+-"]
    pairs = list(zip(keep, row))
  for at, v in pairs:
    if v is None: continue
    if isinstance(v, str):
      s = v.strip()
      v = "?" if s in ("", "?") else ezr.atom(s)
    q[at] = v
  return q

def _fit(tbl: ezr.TBL, q: ezr.ROW) -> ezr.ROW:
  "Q with any non-number in a number column blanked to '?', so distances can be taken."
  out = list(q)
  for at, col in tbl.cols.items():
    if type(col) is ezr.Num and not _is_num(q[at]): out[at] = "?"
  return out

def _surprise(col: ezr.COL, v) -> float:
  "How odd one cell V is for its column COL: 0 is normal, 1 is a likely glitch."
  if col.n == 0: return 0.0                  # training says nothing about this column
  if type(col) is ezr.Sym:
    return 0.0 if v in col.has else 1.0      # a symbol never seen: maybe a typo
  if not _is_num(v): return 1.0              # a number column holding text
  if col.sd == 0:                            # training is constant here
    return 0.0 if v == col.mu else 1.0
  z = abs(v - col.mu) / col.sd
  return min(1.0, max(0.0, z - BAD_Z) / SPAN)

def odd(train_csv, rows) -> list[float]:
  """One suspicion score per row in ROWS (goals included): higher means more
  likely a glitch, a typo, or unlike anything in TRAIN_CSV. Each score is the
  worst cell, plus the mean cell, plus the distance to the nearest training
  row over the input columns."""
  tbl = ezr.Tbl(ezr.csv(train_csv))
  if not tbl.rows:
    return [1.0] * len(rows)                 # nothing to compare with: all unlike it
  known = [_fit(tbl, r) for r in tbl.rows]
  out: list[float] = []
  for row in rows:
    q     = _align(tbl, row)
    cells = [_surprise(col, q[at]) for at, col in tbl.cols.items()
             if q[at] != "?"]
    top   = max(cells, default=0.0)
    mean  = sum(cells) / len(cells) if cells else 0.0
    qf    = _fit(tbl, q)
    near  = (min(ezr.xdist(tbl, qf, s) for s in known)
             if tbl.x else 0.0)              # no input columns: nothing to be far from
    out.append(top + mean + near)
  return out

