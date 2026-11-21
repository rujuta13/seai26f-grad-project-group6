#!/usr/bin/env python3 -B
"step8.py: fix. What is the fewest input flips that make a bad build good?"
from itertools import combinations, product
from ezr import Num, ranker
from step4 import _ref, _row

MAX_VALS  = 8         # values tried per numeric input, spread by rank
MAX_EVALS = 200_000   # tree look-ups allowed for the exact search

_trees: dict = {}     # train_csv -> ranker, grown once per file

def fix(train_csv, row, frozen=()) -> dict:
  """The same row's inputs, with as few values changed as possible so
  that its predicted goals are good.  Columns named in FROZEN never change.
  Exact (fewest changes) while the search is small; greedy beyond that."""
  key = str(train_csv)
  tbl, cutoff = _ref(key)
  if key not in _trees: _trees[key] = ranker(tbl)
  rank  = _trees[key]
  names = tbl.names
  if isinstance(frozen, str): frozen = (frozen,)
  frozen = set(frozen)
  if frozen - set(names):
    raise ValueError(f"frozen columns not in {names}: "
                     f"{sorted(frozen - set(names))}")

  def key_of(v):
    "Numbers first, by value; other symbols after, by text."
    return (0, v, "") if isinstance(v, (int, float)) else (1, 0, str(v))

  def values(col, now) -> list:
    "Values COL could take, from train, other than NOW."
    vals = sorted({r[col.at] for r in tbl.rows} - {"?", now}, key=key_of)
    if type(col) is Num and len(vals) > MAX_VALS:
      vals = [vals[i*len(vals)//MAX_VALS] for i in range(MAX_VALS)]
    return vals

  def out(cand) -> dict:
    "The input cells of CAND, by column name."
    return {s: cand[i] for i, s in enumerate(names) if s[-1] not in "+-"}

  base  = _row(tbl, row)
  moves = []
  for c in tbl.x:
    if names[c.at] in frozen: continue
    vals = values(c, base[c.at])
    if vals: moves.append((c.at, vals))

  if rank(base) <= cutoff: return out(base)

  # Exact: try k changes for k = 1, 2, ... while the search stays small.
  lens = [len(vs) for _, vs in moves]
  ways = [1] + [0]*len(lens)   # ways[k]: ways to pick k moves, one value each
  for n in lens:
    for k in range(len(lens), 0, -1): ways[k] += ways[k-1]*n
  spent = 0
  for k in range(1, len(moves)+1):
    if spent + ways[k] > MAX_EVALS: break
    spent += ways[k]
    best = None
    for combo in combinations(moves, k):
      for picks in product(*(vs for _, vs in combo)):
        cand = list(base)
        for (at, _), v in zip(combo, picks): cand[at] = v
        s = rank(cand)
        if s <= cutoff and (best is None or s < best[0]):
          best = (s, cand)
    if best is not None: return out(best[1])

  # Greedy fallback: flip the one input that most lowers the score, repeatedly.
  cur, cs, free = list(base), rank(base), list(moves)
  while cs > cutoff and free:
    step = None
    for at, vs in free:
      for v in vs:
        cand = list(cur); cand[at] = v
        s = rank(cand)
        if step is None or s < step[0]: step = (s, at, cand)
    if step is None or step[0] >= cs: break
    cs, at, cur = step
    free = [(a, vs) for a, vs in free if a != at]
  return out(cur)

