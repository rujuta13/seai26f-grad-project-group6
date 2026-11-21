#!/usr/bin/env python3 -B
"step1.py: families of flag sets, and the best flag sets."
from ezr import Tbl, csv, xdist, ydist

K   = 5      # a handful of families
TOP = 0.05   # best rows: at most this fraction of the data

def medoid(tbl, members: list) -> list:
  "The member closest, on average, to the others (sampled if big)."
  step = max(1, len(members) // 60)
  pool = members[::step]
  return min(pool, key=lambda r: sum(xdist(tbl, r, z) for z in pool))

def families(train_csv: str) -> list[int]:
  "One family (cluster) label per data row, in file order."
  tbl  = Tbl(csv(train_csv))
  rows = tbl.rows
  n    = len(rows)
  if n == 0: return []
  k = min(K, n)
  # farthest-first seeds: deterministic, so no random draws
  seeds = [0]
  while len(seeds) < k:
    far = max(range(n),
              key=lambda i: min(xdist(tbl, rows[i], rows[j]) for j in seeds))
    seeds.append(far)
  medoids = [rows[i] for i in seeds]
  labels  = [-1] * n
  for _ in range(20):
    new = [min(range(k), key=lambda c: xdist(tbl, r, medoids[c]))
           for r in rows]
    if new == labels: break
    labels = new
    for c in range(k):
      members = [rows[i] for i in range(n) if labels[i] == c]
      if members: medoids[c] = medoid(tbl, members)
  return labels

def best(train_csv: str) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows."
  tbl    = Tbl(csv(train_csv))
  n      = len(tbl.rows)
  ranked = sorted(range(n), key=lambda i: ydist(tbl, tbl.rows[i]))
  return ranked[:int(TOP * n)]

