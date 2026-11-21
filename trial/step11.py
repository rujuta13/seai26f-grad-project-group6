# step11.py
#!/usr/bin/env python3 -B
"Player families (types) and the unusually good rows, via ezr."
import random
import ezr

FAMILIES = 5                # how many player types to sort into
ITERS    = 30               # k-means passes, at most
SKIP     = {"num", "Rank"}  # row ids and ranks derived from OVR: not player types

def _read(train_csv: str) -> tuple[list, list]:
  "Header and data rows of TRAIN_CSV; empty cells become '?', short rows are padded."
  src   = ezr.csv(train_csv)
  first = next(src, None)
  if first is None: return [], []
  head = [str(n) for n in first]
  rows = []
  for r in src:
    r = [("?" if v == "" else v) for v in r]
    rows.append((r + ["?"] * len(head))[:len(head)])
  return head, rows

def _textual(head: list, rows: list) -> set[int]:
  "Columns holding any text other than '?': they cannot be read as numbers."
  return {at for at in range(len(head))
          if any(isinstance(r[at], str) and r[at] != "?" for r in rows)}

def _table(train_csv: str):
  """TRAIN_CSV as an ezr table, or None if the file is empty. Text and SKIP
  columns are renamed to end in X, which ezr ignores; the rest start with a
  capital, so ezr reads them as numbers. Row indices match the file."""
  head, rows = _read(train_csv)
  if not head: return None
  bad   = _textual(head, rows) | {at for at, n in enumerate(head) if n in SKIP}
  names = [(n + "X") if at in bad else (n if n[:1].isupper() else "A " + n)
           for at, n in enumerate(head)]
  return ezr.Tbl([names] + rows)

def _vector(tbl, row: list) -> list[float]:
  "ROW's input columns, each scaled to 0..1 by ezr.norm; unknown is the middle, 0.5."
  return [0.5 if row[c.at] == "?" else ezr.norm(c, row[c.at]) for c in tbl.x]

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

def _gydist(goals: list, row: list) -> float | None:
  "Distance of ROW's goals from the ideal, or None if a goal cell is unknown."
  if any(row[c.at] == "?" for c in goals): return None
  return ezr.dist((abs(ezr.norm(c, row[c.at]) - c.goal) for c in goals),
                  len(goals))

def families(train_csv) -> list[int]:
  "One family (cluster) label per data row of TRAIN_CSV, in file order."
  tbl = _table(train_csv)
  if tbl is None or not tbl.rows: return []
  vs    = [_vector(tbl, r) for r in tbl.rows]
  k     = min(FAMILIES, len(vs))
  rng   = random.Random(ezr.the.Seed)
  labs  = _kmeans(vs, k, rng)
  remap = {}                                   # relabel 0,1,2... by first use
  return [remap.setdefault(c, len(remap)) for c in labs]

def best(train_csv) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows."
  tbl = _table(train_csv)
  if tbl is None or not tbl.rows: return []
  goals = [c for c in tbl.y if c.txt[-1] in "+-"]
  if not goals: return []
  ys    = [_gydist(goals, r) for r in tbl.rows]
  known = [i for i, y in enumerate(ys) if y is not None]
  cap   = len(tbl.rows) * 5 // 100             # at most 5% of all rows
  return sorted(known, key=lambda i: (ys[i], i))[:cap]

