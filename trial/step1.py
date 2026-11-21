# step1.py
import ezr

K = 5  # how many families: a handful

def _nearest(tbl: ezr.TBL, row: ezr.ROW, cents: list) -> int:
  "Index of the centroid closest to ROW, over the x columns."
  return min(range(len(cents)), key=lambda j: ezr.xdist(tbl, row, cents[j]))

def families(train_csv: str) -> list[int]:
  "One family (cluster) label per data row of TRAIN_CSV, in file order."
  tbl  = ezr.Tbl(ezr.csv(train_csv))
  rows = tbl.rows
  if not rows: return []
  k = min(K, len(rows))
  # farthest-point seeding: deterministic, and spreads the seeds out
  cents = [rows[0]]
  while len(cents) < k:
    far = max(range(len(rows)),
              key=lambda i: min(ezr.xdist(tbl, rows[i], c) for c in cents))
    cents.append(rows[far])
  # k-modes style refinement: assign to nearest, then recentre on middles
  labels: list[int] = []
  for _ in range(20):
    new = [_nearest(tbl, r, cents) for r in rows]
    if new == labels: break
    labels = new
    for j in range(k):
      members = [r for r, l in zip(rows, labels) if l == j]
      if members: cents[j] = ezr.mids(ezr.clone(tbl, members))
  return labels

def best(train_csv: str) -> list[int]:
  "Row indices of the best rows on all goals together, at most 5% of rows."
  tbl = ezr.Tbl(ezr.csv(train_csv))
  n   = len(tbl.rows)
  cap = int(0.05 * n)
  return sorted(range(n), key=lambda i: ezr.ydist(tbl, tbl.rows[i]))[:cap]

