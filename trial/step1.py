# step1.py
"""
step1: remember.  Families of flag sets, and the unusually good ones.
Reuses ezr: its table, tree, leaf, and ydist.
"""
import ezr


def families(train_csv: str) -> list[int]:
  "One family (leaf of a small tree) per data row of TRAIN_CSV, in file order."
  tbl = ezr.Tbl(ezr.csv(train_csv))
  old = ezr.the.Leaf
  ezr.the.Leaf = max(4, len(tbl.rows) // 8)   # few, big families
  try:
    node = ezr.tree(tbl, tbl.rows)
  finally:
    ezr.the.Leaf = old                        # always put `the` back
  ids = {id(z): i for i, z in enumerate(ezr.leafs(node))}
  return [ids[id(ezr.leaf(node, r))] for r in tbl.rows]


def best(train_csv: str) -> list[int]:
  "Row indices of the best rows on all goals together; at most 5% of rows."
  tbl  = ezr.Tbl(ezr.csv(train_csv))
  ys   = [ezr.ydist(tbl, r) for r in tbl.rows]   # lower = closer to ideal
  k    = int(0.05 * len(tbl.rows))               # at most 5%
  order = sorted(range(len(tbl.rows)), key=lambda i: ys[i])
  return sorted(order[:k])

