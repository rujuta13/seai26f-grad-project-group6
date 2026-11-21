# step14.py
#!/usr/bin/env python3 -B
"Flow: keep good scouting rows as they arrive; warn when they stop looking like last season's, via ezr."
import math
import random
import ezr

WINDOW = 20      # arrivals judged together against the training baseline
BASE_N = 100     # training rows sampled to build the novelty baseline

_cache: dict[str, tuple] = {}   # train_csv -> (tbl, goals, keep cutoff, inputs, baseline)

def _num(v) -> bool:
  "True for a usable number: an int or float, but not a bool."
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def _cell(v):
  "V as one cell: '?' if blank, None, or not finite; otherwise its atom."
  if v is None: return "?"
  if isinstance(v, str): v = ezr.atom(v.strip() or "?")
  if isinstance(v, float) and not math.isfinite(v): return "?"
  return v

def _align(tbl, row: dict) -> list:
  "Turn a {name: value} dict into a list aligned with tbl.names; a value that cannot be used is '?'."
  out = []
  for at, name in enumerate(tbl.names):
    v   = _cell(row.get(name, "?"))
    col = tbl.cols.get(at)
    if col is not None and type(col) is ezr.Num and not _num(v): v = "?"
    out.append(v)
  return out

def _gydist(goals: list, vals: list):
  "Goal distance of VALS (aligned with the table), or None if a goal cell is not a number."
  if not goals or not all(_num(vals[c.at]) for c in goals): return None
  return ezr.dist((abs(ezr.norm(c, vals[c.at]) - c.goal) for c in goals),
                  len(goals))

def _inputs(tbl) -> list:
  "Input columns that carry signal: not the row id, and not a name unique to each row."
  n = len(tbl.rows)
  return [c for c in tbl.x
          if c.txt != "num"
          and not (type(c) is ezr.Sym and len(c.has) > max(1, n // 2))]

def _dist(xs: list, q: list, r: list) -> float:
  "Distance between aligned rows Q and R over the input columns XS, roughly 0 to 1."
  if not xs: return 0.0
  return ezr.dist((ezr.gap(c, q[c.at], r[c.at]) for c in xs), len(xs))

def _load(train_csv: str) -> tuple:
  "Read TRAIN_CSV once; return its table, goals, keep cutoff, input columns, and novelty baseline."
  if train_csv not in _cache:
    tbl   = ezr.Tbl(ezr.csv(train_csv))
    rows  = tbl.rows
    goals = [c for c in tbl.y if c.txt[-1] in "+-" and type(c) is ezr.Num]
    ys    = sorted(y for y in (_gydist(goals, r) for r in rows) if y is not None)
    k     = max(1, len(ys) * 5 // 100)         # keep the best 5% of training
    cut   = ys[k-1] if ys else None
    xs    = _inputs(tbl)
    base  = []
    if xs and len(rows) > 1:
      rng = random.Random(ezr.the.Seed)        # fixed seed: the same baseline each run
      for i in rng.sample(range(len(rows)), min(BASE_N, len(rows))):
        ds = [_dist(xs, rows[i], rows[j]) for j in range(len(rows)) if j != i]
        base.append(min(ds))                   # leave-one-out nearest distance
    _cache[train_csv] = (tbl, goals, cut, xs, base)
  return _cache[train_csv]

def stream(train_csv, rows) -> list[bool]:
  """Keep (True) each arriving row whose goals are within the best 5% of the
  goal distances in TRAIN_CSV. A row with an unknown goal is not kept. Each
  decision uses only TRAIN_CSV and that row; no arriving row changes the cutoff."""
  tbl, goals, cut, _, _ = _load(train_csv)
  out = []
  for row in rows:
    d = _gydist(goals, _align(tbl, row))
    out.append(cut is not None and d is not None and d <= cut)
  return out

def drift(train_csv, rows) -> int | None:
  """Index of the first row at which the last WINDOW arrivals no longer look like
  TRAIN_CSV (their nearest-training distances differ from the training baseline
  by ezr.same), or None if they never do. Row i is judged from rows 0..i only."""
  tbl, _, _, xs, base = _load(train_csv)
  if not tbl.rows or not xs or len(base) < 2: return None
  window: list[float] = []
  for i, row in enumerate(rows):
    q = _align(tbl, row)
    window.append(min(_dist(xs, q, r) for r in tbl.rows))
    if len(window) > WINDOW: window.pop(0)
    if len(window) == WINDOW and not ezr.same(base, window):
      return i
  return None

