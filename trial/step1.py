# step1.py
import random
import ezr

def families(train_csv: str) -> list[int]:
  "One family (cluster) label per data row of TRAIN_CSV, in file order."
  tbl   = ezr.Tbl(ezr.csv(train_csv))
  rows  = tbl.rows
  n     = len(rows)
  if n == 0: return []
  k     = min(5, n)                        # a handful of families
  rng   = random.Random(ezr.the.Seed)      # reproducible, no global state

  # farthest-point seeding: spread the first centroids out
  cents = [rows[rng.randrange(n)]]
  while len(cents) < k:
    cents.append(max(rows, key=lambda r: min(ezr.xdist(tbl, r, c)
                                             for c in cents)))

  # Lloyd's algorithm over the x columns (the flags)
  labels = [-1] * n
  for _ in range(20):
    new = [min(range(k), key=lambda j, r=r: (ezr.xdist(tbl, r, cents[j]), j))
           for r in rows]
    if new == labels: break
    labels = new
    for j in range(k):
      members = [r for r, l in zip(rows, labels) if l == j]
      if members: cents[j] = ezr.mids(ezr.clone(tbl, members))
  return labels

def best(train_csv: str) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5%."
  tbl   = ezr.Tbl(ezr.csv(train_csv))
  n     = len(tbl.rows)
  cap   = int(0.05 * n)
  score = ezr.wins(tbl)                    # 100 at the best row, 0 at an average one
  ws    = [score(r) for r in tbl.rows]
  ranked = sorted((i for i in range(n) if ws[i] > 0),
                  key=lambda i: -ws[i])
  return ranked[:cap]

