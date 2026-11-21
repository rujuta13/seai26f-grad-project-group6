# step24.py
#!/usr/bin/env python3 -B
"Flow: welcome offers for new customers as they sign up; warn when sign-ups stop looking like the usual customers, via ezr."
import math
import random
import ezr
import step22

KEEP_SHARE = 5     # keep customers in the best this percent of training goal distances
WINDOW     = 20    # arrivals judged together against the training baseline
BASE_N     = 100   # training rows sampled to build the novelty baseline

_cache: dict[str, tuple] = {}   # train_csv -> (tbl, goals, offer cutoff, input columns, baseline)

def _num(v) -> bool:
  "True for a usable number: an int or float, not a bool, and finite."
  return (isinstance(v, (int, float)) and not isinstance(v, bool)
          and math.isfinite(v))

def _cell(v):
  "V as one cell: '?' if blank or unknown, None, or not finite; otherwise its atom."
  if v is None: return "?"
  if isinstance(v, str):
    v = v.strip()
    if v in ("", "?"): return "?"
    v = ezr.atom(v)
  if isinstance(v, float) and not math.isfinite(v): return "?"
  return v

def _align(tbl, row: dict) -> list:
  "ROW's cells aligned with tbl.names; a value a numeric column cannot use is '?'."
  out = []
  for at, name in enumerate(tbl.names):
    v   = _cell(row.get(name, "?"))
    col = tbl.cols.get(at)
    if type(col) is ezr.Num and not _num(v): v = "?"
    out.append(v)
  return out

def _goals(tbl) -> list:
  "The goal columns: numeric names ending in + or -."
  return [c for c in tbl.y if c.txt[-1] in "+-" and type(c) is ezr.Num]

def _gydist(goals: list, vals: list) -> float | None:
  "Goal distance of VALS from the ideal, or None if a goal is not a number."
  if not goals or not all(_num(vals[c.at]) for c in goals): return None
  return ezr.dist((abs(ezr.norm(c, vals[c.at]) - c.goal) for c in goals),
                  len(goals))

def _inputs(tbl) -> list:
  "Input columns that carry signal: not the row id, and not a text column unique to each row."
  n = len(tbl.rows)
  return [c for c in tbl.x
          if c.txt != "num"
          and not (type(c) is ezr.Sym and len(c.has) > max(1, n // 2))]

def _dist(xs: list, q: list, r: list) -> float:
  "Distance between aligned rows Q and R over the input columns XS, roughly 0 to 1."
  if not xs: return 0.0
  return ezr.dist((ezr.gap(c, q[c.at], r[c.at]) for c in xs), len(xs))

def _load(train_csv: str) -> tuple:
  "Read TRAIN_CSV once; return its table, goals, offer cutoff, input columns, and novelty baseline."
  if train_csv not in _cache:
    tbl   = ezr.Tbl(ezr.csv(train_csv))
    rows  = tbl.rows
    goals = _goals(tbl)
    ys    = sorted(y for y in (_gydist(goals, r) for r in rows) if y is not None)
    k     = max(1, len(ys) * KEEP_SHARE // 100)
    cut   = ys[k - 1] if ys else None
    xs    = _inputs(tbl)
    base  = []
    if xs and len(rows) > 1:
      rng = random.Random(ezr.the.Seed)          # fixed seed: the same baseline each run
      for i in rng.sample(range(len(rows)), min(BASE_N, len(rows))):
        ds = [_dist(xs, rows[i], rows[j]) for j in range(len(rows)) if j != i]
        base.append(min(ds))                     # leave-one-out nearest distance
    _cache[train_csv] = (tbl, goals, cut, xs, base)
  return _cache[train_csv]

def _fill(train_csv: str, goals: list, row: dict, vals: list) -> list:
  "VALS with each unknown goal estimated from ROW's inputs by step22 (the median of its nearest training rows)."
  if all(_num(vals[c.at]) for c in goals): return vals
  names  = {c.txt for c in goals}
  inputs = {k: v for k, v in row.items() if k not in names}
  est    = step22.guess(train_csv, inputs)
  out    = list(vals)
  for c in goals:
    if not _num(out[c.at]) and c.txt in est:
      out[c.at] = float(est[c.txt][0])
  return out

def stream(train_csv, rows) -> list[bool]:
  """Keep (True) each arriving row worth a welcome offer: its goals, known or
  estimated from its inputs, sit within the best KEEP_SHARE percent of the goal
  distances in TRAIN_CSV. A row whose goals cannot be judged is not kept. Each
  decision uses only TRAIN_CSV and that row; no arrival changes the cutoff."""
  tbl, goals, cut, _, _ = _load(train_csv)
  out = []
  for row in rows:
    vals = _align(tbl, row)
    if cut is not None: vals = _fill(train_csv, goals, row, vals)
    d = _gydist(goals, vals)
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

