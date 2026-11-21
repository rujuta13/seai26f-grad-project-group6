"step1.py: families of flag combinations, and the best of them."
import random
import ezr

K      = 5    # how many families: a handful
ROUNDS = 20   # max refinement rounds for the families

def families(train_csv: str) -> list[int]:
  "One family (cluster) label per data row of TRAIN_CSV, in file order."
  tbl  = ezr.Tbl(ezr.csv(train_csv))
  rows = tbl.rows
  n    = len(rows)
  if n == 0: return []
  k    = min(K, n)
  rng  = random.Random(ezr.the.Seed)
  # farthest-first seeds: start from a random row, then keep adding
  # the row furthest from every seed chosen so far
  seeds = [rows[rng.randrange(n)]]
  far   = [ezr.xdist(tbl, r, seeds[0]) for r in rows]
  while len(seeds) < k:
    i   = max(range(n), key=lambda j: far[j])
    seeds.append(rows[i])
    far = [min(g, ezr.xdist(tbl, r, rows[i])) for g, r in zip(far, rows)]
  # k-means style: assign each row to its nearest seed, then replace
  # each seed by its cluster's middle (a MIDS centroid)
  labels = None
  for _ in range(ROUNDS):
    new = [min(range(k), key=lambda c: ezr.xdist(tbl, r, seeds[c]))
           for r in rows]
    if new == labels: break
    labels = new
    for c in range(k):
      members = [r for r, l in zip(rows, labels) if l == c]
      if members:
        seeds[c] = ezr.mids(ezr.clone(tbl, members))
  return labels

def best(train_csv: str) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows."
  tbl    = ezr.Tbl(ezr.csv(train_csv))
  rows   = tbl.rows
  cap    = len(rows) * 5 // 100       # at most 5%, in whole rows
  ranked = sorted(range(len(rows)), key=lambda i: ezr.ydist(tbl, rows[i]))
  return ranked[:cap]

