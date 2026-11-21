# step1.py
"""Families of flag sets, and the unusually good ones.

Families are k-medoids clusters over the flag columns (the x columns).
Good rows are those closest to the ideal on all goals together.
"""
import ezr


def families(train_csv: str, k: int = 5) -> list[int]:
  "One family (cluster) label per data row of TRAIN_CSV, in file order."
  tbl  = ezr.Tbl(ezr.csv(train_csv))
  rows = tbl.rows
  n    = len(rows)
  k    = max(1, min(k, n))

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
    return min(group, key=lambda a: sum(ezr.xdist(tbl, rows[a], rows[b])
                                        for b in group))

  meds   = [rows[i] for i in seeds]
  labels = assign(meds)
  for _ in range(3):
    for c in range(k):
      group = [i for i in range(n) if labels[i] == c]
      if group: meds[c] = rows[medoid(group)]
    new = assign(meds)
    if new == labels: break
    labels = new
  return labels


def best(train_csv: str) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows."
  tbl  = ezr.Tbl(ezr.csv(train_csv))
  rows = tbl.rows
  n    = len(rows)
  cap  = max(1, n // 20)              # floor of 5%, so never above 5%
  order = sorted(range(n), key=lambda i: ezr.ydist(tbl, rows[i]))
  return order[:cap]

