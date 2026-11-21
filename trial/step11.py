# step11.py
"""Families and good rows, scaled to about 17,000 players.

families(): k-medoids over the x columns, as step1.families does, but
each family's medoid is picked from a sample of its members. Cost grows
with the number of rows, not with its square.

best(): the unusually good rows, straight from step1.best.
"""
import random

import ezr
import step1

K      = 5     # families (clusters)
CANDS  = 40    # candidate medoids tried per family
REFS   = 100   # family members each candidate is scored against
PASSES = 3     # assign / re-medoid rounds


def families(train_csv: str) -> list[int]:
  "One family (cluster) label per data row of TRAIN_CSV, in file order."
  tbl  = ezr.Tbl(ezr.csv(train_csv))
  rows = tbl.rows
  n    = len(rows)
  k    = max(1, min(K, n))
  rng  = random.Random(ezr.the.Seed)

  # farthest-first seeds, deterministic, starting from row 0
  seeds = [0]
  d = [ezr.xdist(tbl, r, rows[0]) for r in rows]
  while len(seeds) < k:
    i = max(range(n), key=lambda j: d[j])
    seeds.append(i)
    d = [min(d[j], ezr.xdist(tbl, rows[j], rows[i])) for j in range(n)]

  def assign(meds: list) -> list[int]:
    return [min(range(len(meds)),
                key=lambda c: ezr.xdist(tbl, r, meds[c]))
            for r in rows]

  def medoid(group: list[int]) -> int:
    "The member closest, on average, to a sample of its family."
    cands = group if len(group) <= CANDS else rng.sample(group, CANDS)
    refs  = group if len(group) <= REFS  else rng.sample(group, REFS)
    return min(cands,
               key=lambda a: sum(ezr.xdist(tbl, rows[a], rows[b])
                                 for b in refs))

  meds   = [rows[i] for i in seeds]
  labels = assign(meds)
  for _ in range(PASSES):
    for c in range(k):
      group = [i for i in range(n) if labels[i] == c]
      if group: meds[c] = rows[medoid(group)]
    new = assign(meds)
    if new == labels: break
    labels = new
  return labels


def best(train_csv: str) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows."
  return step1.best(train_csv)

