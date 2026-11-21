# step4.py
#!/usr/bin/env python3 -B
"Flow: keep good nightly rows as they arrive; warn when they stop looking like training, via ezr."
import random
import ezr

_cache: dict[str, tuple] = {}   # train_csv -> (tbl, keep cutoff, novelty baseline)

def _ydist(tbl, row: list) -> float | None:
  "ydist of ROW (aligned with tbl.names), or None if a goal cell is unknown."
  if any(row[c.at] == "?" for c in tbl.y): return None
  return ezr.ydist(tbl, row)

def _align(tbl, row: dict) -> list:
  "Turn a {name: value} dict into a list aligned with tbl.names."
  out = []
  for name in tbl.names:
    v = row.get(name, "?")
    if isinstance(v, str): v = ezr.atom(v)
    out.append(v)
  return out

def _load(train_csv: str) -> tuple:
  "Read TRAIN_CSV once; return its table, the keep cutoff, and a novelty baseline."
  if train_csv not in _cache:
    tbl  = ezr.Tbl(ezr.csv(train_csv))
    rows = tbl.rows
    # keep cutoff: the best 5% of training rows on all goals together
    ys   = sorted(y for y in (_ydist(tbl, r) for r in rows) if y is not None)
    k    = max(1, len(ys) * 5 // 100)
    cut  = ys[k-1] if ys else None
    # novelty baseline: each sampled training row's distance to its nearest
    # other training row (leave-one-out), over the x columns
    rng  = random.Random(ezr.the.Seed)
    idx  = rng.sample(range(len(rows)), min(100, len(rows)))
    base = []
    for i in idx:
      ds = [ezr.xdist(tbl, rows[i], rows[j])
            for j in range(len(rows)) if j != i]
      if ds: base.append(min(ds))
    _cache[train_csv] = (tbl, cut, base)
  return _cache[train_csv]

def stream(train_csv, rows) -> list[bool]:
  """Keep (True) each arriving row whose goals are within the best 5% seen
  in TRAIN_CSV. Only the training table is used; no row is ever added to it."""
  tbl, cut, _ = _load(train_csv)
  out = []
  for row in rows:
    y = _ydist(tbl, _align(tbl, row))
    out.append(cut is not None and y is not None and y <= cut)
  return out

def drift(train_csv, rows) -> int | None:
  """Index of the first row at which the last WINDOW arrivals no longer look
  like TRAIN_CSV (their nearest-training distances differ from the training
  baseline by ezr.same), or None if they never do."""
  tbl, _, base = _load(train_csv)
  if not tbl.rows or len(base) < 2: return None
  WINDOW = 20
  window: list[float] = []
  for i, row in enumerate(rows):
    q = _align(tbl, row)
    window.append(min(ezr.xdist(tbl, q, r) for r in tbl.rows))
    if len(window) > WINDOW: window.pop(0)
    if len(window) == WINDOW and not ezr.same(base, window):
      return i
  return None

