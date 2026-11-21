# step1.py
"""Step 1: families of flag combinations, and the unusually good rows.

Uses only the standard library plus ezr.
"""
import random
import ezr


def _load(train_csv: str) -> ezr.TBL:
  "Read TRAIN_CSV into an ezr table, keeping file order."
  return ezr.Tbl(ezr.csv(train_csv))


def families(train_csv: str) -> list[int]:
  "One family label per data row of TRAIN_CSV, in file order."
  random.seed(ezr.the.Seed)
  tbl  = _load(train_csv)
  rows = tbl.rows
  if not rows: return []
  k = min(5, len(rows))                       # a handful of families
  # Clustering uses the flags only (x columns): k-modes/k-means style,
  # with centroids made by ezr.mids and distances by ezr.xdist.
  cents  = random.sample(rows, k)             # start from real rows
  labels = None
  for _ in range(20):
    new = [min(range(k), key=lambda j: ezr.xdist(tbl, r, cents[j]))
           for r in rows]
    if new == labels: break                   # assignments are stable
    labels = new
    for j in range(k):
      members = [r for r, l in zip(rows, labels) if l == j]
      if members:                             # keep old centroid if empty
        cents[j] = ezr.mids(ezr.clone(tbl, members))
  return labels


def best(train_csv: str) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows."
  tbl  = _load(train_csv)
  rows = tbl.rows
  cap  = int(0.05 * len(rows))                # at most 5% of rows
  if cap <= 0: return []

  def gaps(r: ezr.ROW) -> list[float]:
    "Per goal: distance from the ideal. Smaller is better."
    return [abs(ezr.norm(c, r[c.at]) - c.goal) for c in tbl.y]

  G = [gaps(r) for r in rows]

  def dominates(a: list[float], b: list[float]) -> bool:
    "A is no worse than B on every goal, and strictly better on one."
    return (all(x <= y for x, y in zip(a, b))
            and any(x < y for x, y in zip(a, b)))

  # Pareto front: rows that no other row beats on all goals together.
  front = [i for i in range(len(rows))
           if not any(dominates(G[j], G[i])
                      for j in range(len(rows)) if j != i)]
  # Keep the closest to the ideal, capped at 5%; return in file order.
  front.sort(key=lambda i: ezr.ydist(tbl, rows[i]))
  return sorted(front[:cap])

