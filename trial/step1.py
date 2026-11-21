# step1.py
#!/usr/bin/env python3 -B
"""
step1.py: remember.  Families of flag sets, and the best of them.
Reuses ezr for tables, trees, and distance.
"""
import ezr

def families(train_csv: str) -> list[int]:
  "One family (leaf of a tree grown on the rows) per data row, in file order."
  tbl = ezr.Tbl(ezr.csv(train_csv))
  old = ezr.the.Leaf
  ezr.the.Leaf = max(old, len(tbl.rows)//8)   # few, big families
  try:     node = ezr.tree(tbl, tbl.rows)
  finally: ezr.the.Leaf = old
  ids = {id(z): i for i, z in enumerate(ezr.leafs(node))}
  return [ids[id(ezr.leaf(node, r))] for r in tbl.rows]

def best(train_csv: str) -> list[int]:
  "Row indices of the best rows on all goals together: at most 5% of rows."
  tbl   = ezr.Tbl(ezr.csv(train_csv))
  n     = int(0.05 * len(tbl.rows))
  order = sorted(range(len(tbl.rows)),
                 key=lambda i: ezr.ydist(tbl, tbl.rows[i]))
  return order[:n]

