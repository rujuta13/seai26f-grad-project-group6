# step1.py
"Step 1: families (clusters of flag sets) and best (unusually good rows)."
import random
import ezr

K     = 5             # number of families to find
ITERS = 20            # cap on k-modes iterations
SEED  = 1234567891    # fixed seed, so results repeat
FRAC  = 0.05          # "best" is at most this fraction of rows

def _load(train_csv: str):
  "Table of TRAIN_CSV, with columns and goals from its header."
  return ezr.Tbl(ezr.csv(train_csv))

def families(train_csv: str) -> list[int]:
  "One family (cluster) label per data row of TRAIN_CSV, in file order."
  tbl  = _load(train_csv)
  rows = tbl.rows
  if not rows: return []
  k     = min(K, len(rows))
  rng   = random.Random(SEED)
  cents = [list(r) for r in rng.sample(rows, k)]   # seed centroids with rows
  labels = [-1] * len(rows)
  for _ in range(ITERS):
    # assign each row to its nearest centroid, over the input (x) columns
    new = [min(range(k), key=lambda j, r=r: ezr.xdist(tbl, r, cents[j]))
           for r in rows]
    if new == labels: break
    labels = new
    # move each centroid to the middle (mode per symbol) of its members
    for j in range(k):
      members = [r for r, lab in zip(rows, labels) if lab == j]
      if members:
        cents[j] = ezr.mids(ezr.clone(tbl, members))
  return labels

def best(train_csv: str) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows, best first."
  tbl   = _load(train_csv)
  rows  = tbl.rows
  n     = int(FRAC * len(rows))
  order = sorted(range(len(rows)), key=lambda i: ezr.ydist(tbl, rows[i]))
  return order[:n]

