# step2.py
#!/usr/bin/env python3 -B
"Guess each goal of a flag set from its nearest labelled neighbours, via ezr."
import ezr

_cache: dict[str, tuple] = {}     # train_csv -> (tbl, goal columns)

def _load(train_csv: str) -> tuple:
  "Read TRAIN_CSV once; return its table and its goal columns."
  if train_csv not in _cache:
    tbl   = ezr.Tbl(ezr.csv(train_csv))
    goals = [c for c in tbl.y if c.txt[-1] in "+-"]
    _cache[train_csv] = (tbl, goals)
  return _cache[train_csv]

def _pct(vs: list, p: float) -> float:
  "The value at fraction P of the sorted VS (nearest rank)."
  vs = sorted(vs)
  return vs[round(p * (len(vs) - 1))]

def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
  """Estimate each goal of ROW (a dict with its goal cells removed).
  The estimate is the median goal value of the nearest labelled rows;
  low..high is the 10th..90th percentile of those same neighbours,
  so a wide range means the neighbours disagree."""
  tbl, goals = _load(train_csv)
  rows = tbl.rows
  q    = [row.get(name, "?") for name in tbl.names]  # aligned with columns
  k    = max(3, min(len(rows), int(len(rows) ** 0.5)))
  near = sorted(rows, key=lambda r: ezr.xdist(tbl, q, r))[:k]
  out  = {}
  for g in goals:
    vs = [float(r[g.at]) for r in near if r[g.at] != "?"]
    if not vs:                       # no neighbour has this goal: use all rows
      vs = [float(r[g.at]) for r in rows if r[g.at] != "?"]
    if not vs:
      out[g.txt] = (0.0, 0.0, 0.0)
      continue
    out[g.txt] = (_pct(vs, 0.5), _pct(vs, 0.1), _pct(vs, 0.9))
  return out

