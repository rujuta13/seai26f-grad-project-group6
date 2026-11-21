# step1.py
#!/usr/bin/env python3 -B
"Families and best rows of a flag-combination table, via ezr."
import random
import ezr

def families(train_csv: str) -> list[int]:
  "One family (cluster) label per data row of TRAIN_CSV, in file order."
  tbl  = ezr.Tbl(ezr.csv(train_csv))
  rows = tbl.rows
  n    = len(rows)
  if n == 0: return []
  k    = min(5, n)                              # a handful of families
  cap  = 40                                     # medoid update: sample size
  rng  = random.Random(ezr.the.Seed)
  dist = lambda i, j: ezr.xdist(tbl, rows[i], rows[j])  # x columns only
  # farthest-first seeds, so the families start out spread apart
  meds = [rng.randrange(n)]
  while len(meds) < k:
    meds.append(max(range(n), key=lambda i: min(dist(i, m) for m in meds)))
  assign = lambda ms: [min(range(k), key=lambda c: dist(i, ms[c]))
                       for i in range(n)]
  labs = assign(meds)
  for _ in range(20):                           # k-medoids, until stable
    new = []
    for c in range(k):
      mem = [i for i in range(n) if labs[i] == c]
      if not mem: new.append(meds[c]); continue
      pool = mem if len(mem) <= cap else rng.sample(mem, cap)
      new.append(min(pool, key=lambda z: sum(dist(z, o) for o in pool)))
    if new == meds: break
    meds = new
    labs = assign(meds)
  remap = {}                                    # relabel 0,1,2... by first use
  return [remap.setdefault(c, len(remap)) for c in labs]

def best(train_csv: str) -> list[int]:
  "Row indices of the best rows on all goals together: at most 5% of rows."
  tbl  = ezr.Tbl(ezr.csv(train_csv))
  rows = tbl.rows
  ys   = [ezr.ydist(tbl, r) for r in rows]     # distance from ideal goals
  cap  = len(rows) * 5 // 100
  return sorted(range(len(rows)), key=lambda i: ys[i])[:cap]

