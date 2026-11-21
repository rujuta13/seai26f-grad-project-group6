# step9.py
"""
step9: odd.  Flag benchmark results that look like glitches, typos, or
flag sets unlike anything in training.  Each row gets one suspicion
score; higher means more odd.  Two parts, summed:
  cells: each cell is judged against training's stats for its own column.
         Text, inf, or nan in a number column is a glitch; a wild number
         is far from the column's mean; a symbol training never saw is odd.
  shape: how far the row's inputs sit from the nearest training row,
         scaled by how far training rows usually sit from one another.
Reuses step2 (model, _full) and ezr (xdist, dist, Num, Sym).
"""
import math
import ezr
import step2

CAP   = 10.0          # no single check adds more than this
_base: dict = {}      # train_csv -> typical nearest-neighbour distance


def _finite(v) -> bool:
  "True for a real, finite number."
  return isinstance(v, (int, float)) and math.isfinite(v)


def _clean(tbl, full: list) -> list:
  "FULL, with any cell the distance code cannot read set to '?'."
  out = list(full)
  for c in tbl.cols.values():
    v = out[c.at]
    if type(c) is ezr.Num and v != "?" and not _finite(v):
      out[c.at] = "?"
  return out


def _cell(c, v) -> float:
  "How odd one cell V is, judged against training's own column C."
  if v == "?": return 0.0                    # unknown is not a glitch
  if type(c) is ezr.Sym:
    k = c.has.get(v, 0)
    if k == 0: return CAP / 2                # a value training never saw
    return min(CAP / 2, max(0.0, -math.log2(k / c.n) - 3))
  if not _finite(v): return CAP              # text, inf, or nan
  z = abs(v - c.mu) / (c.sd + 1e-32)
  return min(CAP, max(0.0, z - 3))


def _nn_base(train_csv: str) -> float:
  """Typical nearest-neighbour distance among training rows: the 95th
  percentile over a sample of about 100 rows, never below one mismatch."""
  if train_csv not in _base:
    tbl, _ = step2._model(train_csv)
    rows   = [_clean(tbl, r) for r in tbl.rows]
    n      = len(rows)
    step   = max(1, n // 100)
    near   = sorted(min((ezr.xdist(tbl, rows[i], rows[j])
                         for j in range(n) if j != i), default=0.0)
                    for i in range(0, n, step))
    q95    = near[min(len(near) - 1, int(0.95 * len(near)))] if near else 0.0
    _base[train_csv] = max(q95, ezr.dist([1], max(1, len(tbl.x))))
  return _base[train_csv]


def odd(train_csv, rows) -> list[float]:
  """One suspicion score per row in ROWS (goals included): higher is more
  likely a glitch, a typo, or unlike anything in TRAIN_CSV."""
  tbl, _ = step2._model(train_csv)
  if not tbl.rows:
    raise ValueError("odd: no training rows to compare against")
  base  = _nn_base(train_csv)
  train = [_clean(tbl, r) for r in tbl.rows]
  out   = []
  for row in rows:
    full  = step2._full(tbl, row)            # goals kept, so glitches show
    clean = _clean(tbl, full)                # safe for distance code
    cells = sum(_cell(c, full[c.at]) for c in tbl.cols.values())
    near  = min(ezr.xdist(tbl, clean, r) for r in train)
    shape = min(CAP, max(0.0, near / base - 1))
    out.append(float(cells + shape))
  return out

