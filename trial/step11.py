# step11.py: remember. Player types (families), and the players who are strong and good from the spot.
"step11.py: remember. Player types (families), and the players who are strong and good from the spot."
import math
import random
import ezr

K      = 5    # how many player types: a handful
ROUNDS = 20   # max refinement rounds for the types

def _load(train_csv):
  "The table, its input columns, and each row as one entry per input: 0..1 for nums, the symbol for syms, None if unknown."
  tbl = ezr.Tbl(ezr.csv(train_csv))
  cs  = list(tbl.x)
  vs  = [[None if r[c.at] == "?" else
          r[c.at] if type(c) is ezr.Sym else ezr.norm(c, r[c.at])
          for c in cs] for r in tbl.rows]
  return tbl, cs, vs

def _gap2(cs, a, b):
  "Squared distance between two entry-lists A and B, column by column; unknown is far."
  total = 0
  for c, x, y in zip(cs, a, b):
    if x is None or y is None: g = 1
    elif type(c) is ezr.Sym:   g = 0 if x == y else 1
    else:                      g = abs(x - y)
    total += g * g
  return total

def _centre(cs, members):
  "A centroid of MEMBERS: the mean for numbers, the most common for symbols; None where none is known."
  out = []
  for j, c in enumerate(cs):
    xs = [m[j] for m in members if m[j] is not None]
    if not xs: out.append(None)
    elif type(c) is ezr.Sym:
      counts = {}
      for x in xs: counts[x] = counts.get(x, 0) + 1
      out.append(max(counts, key=counts.get))
    else:
      out.append(sum(xs) / len(xs))
  return out

def families(train_csv: str) -> list[int]:
  "One family (cluster) label per data row of TRAIN_CSV, in file order."
  tbl, cs, vs = _load(train_csv)
  n = len(vs)
  if n == 0: return []
  k   = min(K, n)
  rng = random.Random(ezr.the.Seed)
  # farthest-first seeds: start from a random row, then keep adding
  # the row furthest from every seed chosen so far
  seeds = [vs[rng.randrange(n)]]
  far   = [_gap2(cs, v, seeds[0]) for v in vs]
  while len(seeds) < k:
    i = max(range(n), key=lambda j: far[j])
    seeds.append(vs[i])
    far = [min(g, _gap2(cs, v, vs[i])) for g, v in zip(far, vs)]
  # k-means style: assign each row to its nearest seed, then replace
  # each seed by its cluster's centroid
  labels = None
  for _ in range(ROUNDS):
    new = [min(range(k), key=lambda c: _gap2(cs, v, seeds[c])) for v in vs]
    if new == labels: break
    labels = new
    for c in range(k):
      members = [v for v, l in zip(vs, labels) if l == c]
      if members: seeds[c] = _centre(cs, members)
  return labels

def best(train_csv: str) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows."
  tbl = ezr.Tbl(ezr.csv(train_csv))
  if not tbl.y: raise ValueError("no goal columns (names ending in + or -)")
  rows = tbl.rows
  cap  = len(rows) * 5 // 100          # at most 5%, in whole rows

  def ydist_or_inf(r):
    "How far ROW's goals are from ideal; infinite if any goal is unknown."
    if any(r[c.at] == "?" for c in tbl.y): return math.inf
    return ezr.ydist(tbl, r)

  ys     = [ydist_or_inf(r) for r in rows]
  ranked = sorted(range(len(rows)), key=lambda i: ys[i])
  known  = [i for i in ranked if ys[i] < math.inf]   # rows with unknown goals cannot be called good
  return known[:cap]

