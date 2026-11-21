# step2.py
"""
step2: guess.  Given a flag set with its goal cells removed, guess each
goal as (estimate, low, high).  Reuses ezr's table, tree, leaf, and atom.
The estimate is the median goal value of the training rows in the same
leaf; low..high is the 10th..90th percentile of those rows.
"""
import ezr

_cache: dict = {}   # train_csv -> (tbl, node), so the tree is grown once


def _model(train_csv: str):
  "Grow one tree on the training rows (once per file). Return (tbl, node)."
  if train_csv not in _cache:
    tbl = ezr.Tbl(ezr.csv(train_csv))
    old = ezr.the.Leaf
    ezr.the.Leaf = max(4, len(tbl.rows) // 16)   # big enough families
    try:
      node = ezr.tree(tbl, tbl.rows)
    finally:
      ezr.the.Leaf = old                          # always put `the` back
    _cache[train_csv] = (tbl, node)
  return _cache[train_csv]


def _cell(v):
  "Strings become ATOMs (so '1' -> 1); numbers and '?' pass through."
  return ezr.atom(v) if isinstance(v, str) else v


def _full(tbl, row) -> list:
  """Put ROW back into full table order. Goal cells are unknown ('?').
  ROW may be a dict {column name: value} (the usual convention) or a list.
  A list is either full-length, or has only the non-goal cells."""
  names = tbl.names
  if isinstance(row, dict):
    return [_cell(row[n]) if n in row else "?" for n in names]
  if len(row) == len(names):
    return [_cell(v) for v in row]
  goals = {c.at for c in tbl.y}
  xs = iter(row)
  return ["?" if at in goals else _cell(next(xs))
          for at in range(len(names))]


def _q(vs: list, p: float) -> float:
  "The p-th quantile (nearest rank) of sorted VS."
  return float(vs[min(len(vs) - 1, int(p * len(vs)))])


def guess(train_csv: str, row) -> dict[str, tuple[float, float, float]]:
  """For every goal: (median, 10th percentile, 90th percentile) of that
  goal's values among the training rows that share ROW's leaf.
  Keys are the goal column names, as in the csv header."""
  tbl, node = _model(train_csv)
  fam = ezr.leaf(node, _full(tbl, row)).rows
  out = {}
  for c in tbl.y:
    vs = sorted(r[c.at] for r in fam if r[c.at] != "?")
    if len(vs) < 5:                  # family too thin: use every row
      vs = sorted(r[c.at] for r in tbl.rows if r[c.at] != "?")
    if not vs:
      out[c.txt] = (float("nan"),) * 3
    else:
      out[c.txt] = (_q(vs, 0.5), _q(vs, 0.1), _q(vs, 0.9))
  return out

