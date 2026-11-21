# step21.py
#!/usr/bin/env python3 -B
"Kinds of customer (families) and the unusually good customers, via ezr."
import math
import random
import ezr

FAMILIES   = 5    # how many kinds of shopper to sort customers into
ITERS      = 30   # k-means passes, at most
BEST_SHARE = 5    # the unusually good rows: at most this percent of all rows

def _num(v) -> bool:
  "True for an int or float, but not a bool."
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def _cell(v):
  "V as one cell: '?' if blank or unknown, or not finite; otherwise its atom."
  if v is None: return "?"
  if isinstance(v, str):
    v = v.strip()
    if v in ("", "?"): return "?"
    v = ezr.atom(v)
  if isinstance(v, float) and not math.isfinite(v): return "?"
  return v

def _is_goal(name: str) -> bool:
  "True for a goal column: its name ends in + or -."
  return name[-1:] in ("+", "-")

def _is_input(name: str) -> bool:
  "True for an input column: neither a goal nor an ignored X column."
  return name[-1:] not in ("+", "-", "X")

def _read(train_csv: str) -> tuple[list, list]:
  "Header and data rows of TRAIN_CSV, cells read by _cell; short rows are padded with '?'."
  src   = ezr.csv(train_csv)
  first = next(src, None)
  if first is None: return [], []
  head  = [str(v).strip() or f"col{at}" for at, v in enumerate(first)]
  width = len(head)
  rows  = []
  for r in src:
    cells = [_cell(v) for v in r][:width]
    rows.append(cells + ["?"] * (width - len(cells)))
  return head, rows

def _numeric_cols(head: list, rows: list, want) -> list[int]:
  "Positions of columns that WANT selects and that are mostly numbers."
  out = []
  for at, name in enumerate(head):
    if not want(name): continue
    vals = [r[at] for r in rows if r[at] != "?"]
    if vals and sum(_num(v) for v in vals) * 2 >= len(vals):
      out.append(at)
  return out

def _scaled(head: list, rows: list, ats: list[int]) -> list[list[float]]:
  "Each row's cells in columns ATS, scaled to 0..1 by ezr.norm; a missing or non-number cell is the middle, 0.5."
  cols = {at: ezr.Num(head[at], at) for at in ats}
  for r in rows:
    for at in ats:
      if _num(r[at]): ezr.add(cols[at], r[at])
  return [[ezr.norm(cols[at], r[at]) if _num(r[at]) else 0.5 for at in ats]
          for r in rows]

def _sq(a: list, b: list) -> float:
  "Squared distance between two feature vectors."
  return sum((x - y) ** 2 for x, y in zip(a, b))

def _kmeans(vs: list, k: int, rng: random.Random) -> list[int]:
  "Lloyd's k-means with farthest-first starts, so the families start out spread apart."
  n     = len(vs)
  cents = [list(vs[rng.randrange(n)])]
  while len(cents) < k:
    far = max(range(n), key=lambda i: min(_sq(vs[i], c) for c in cents))
    cents.append(list(vs[far]))
  labs = []
  for _ in range(ITERS):
    new = [min(range(k), key=lambda c: _sq(v, cents[c])) for v in vs]
    if new == labs: break
    labs = new
    for c in range(k):
      mem = [vs[i] for i in range(n) if labs[i] == c]
      cents[c] = ([sum(col) / len(mem) for col in zip(*mem)] if mem
                  else list(vs[rng.randrange(n)]))
  return labs

def families(train_csv) -> list[int]:
  "One family (kind of shopper) label per data row of TRAIN_CSV, in file order."
  head, rows = _read(train_csv)
  if not rows: return []
  ats = _numeric_cols(head, rows, _is_input)
  if not ats: return [0] * len(rows)
  vs    = _scaled(head, rows, ats)
  k     = min(FAMILIES, len(vs))
  rng   = random.Random(ezr.the.Seed)
  labs  = _kmeans(vs, k, rng)
  remap = {}                                   # relabel 0,1,2... by first use
  return [remap.setdefault(c, len(remap)) for c in labs]

def best(train_csv) -> list[int]:
  "Row indices of the unusually good customers (best on all goals together), at most 5% of rows."
  head, rows = _read(train_csv)
  if not rows: return []
  gats = _numeric_cols(head, rows, _is_goal)
  if not gats: return []
  cols = {at: ezr.Num(head[at], at) for at in gats}
  for r in rows:
    for at in gats:
      if _num(r[at]): ezr.add(cols[at], r[at])
  ys = []
  for r in rows:
    if any(not _num(r[at]) for at in gats):    # an unknown goal: not ranked
      ys.append(None)
      continue
    ys.append(ezr.dist((abs(ezr.norm(cols[at], r[at]) - cols[at].goal)
                        for at in gats), len(gats)))
  known = [i for i, y in enumerate(ys) if y is not None]
  cap   = len(rows) * BEST_SHARE // 100
  return sorted(known, key=lambda i: (ys[i], i))[:cap]

