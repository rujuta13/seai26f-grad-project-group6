# step11.py
"""Step 11: remember.  Sort players into the types they really are, and
show the players who are both strong and good from the spot.

Identifier and text columns (num, Rank, Name, Nation, League, Team, url)
are left out, so they cannot drive the clusters or the ranking.  Numeric
columns keep only numeric cells; anything else counts as unknown.

Uses only the standard library plus ezr.
"""
import math
import random
from collections import Counter
import ezr

_K      = 5                                   # how many player types to look for
_ROUNDS = 20                                  # most passes of the family search
_CUT    = 0.05                                # best-on-all-goals: at most this share
_DROP   = {"num", "Rank", "Name", "Nation", "League", "Team", "url"}


def _load(train_csv: str) -> ezr.TBL:
  "Read TRAIN_CSV into an ezr table, dropping identifier and text columns."
  raw = list(ezr.csv(train_csv))
  if not raw: return ezr.Tbl([[]])
  head = [str(h).strip() for h in raw[0]]
  keep = [i for i, h in enumerate(head)
          if h not in _DROP and not h.endswith("X")]
  rows = []
  for r in raw[1:]:
    r = list(r) + ["?"] * (len(head) - len(r))   # pad ragged rows
    row = []
    for i in keep:
      v = r[i]
      if head[i][:1].isupper() and not isinstance(v, (int, float)):
        v = "?"                                  # unreadable numeric cell
      row.append(v)
    rows.append(row)
  return ezr.Tbl([[head[i] for i in keep]] + rows)


def _encode(tbl: ezr.TBL, rows: ezr.ROWS) -> list[list]:
  "Each row's input cells: numbers scaled to 0..1, symbols as-is, None if unknown."
  return [[None if r[c.at] == "?" else ezr.norm(c, r[c.at])
           for c in tbl.x] for r in rows]


def _dist(a: list, b: list, sym: list[bool]) -> float:
  "Root mean square gap between two encoded input rows, as ezr.xdist."
  s = 0.0
  for x, y, is_sym in zip(a, b, sym):
    if x is None or y is None: g = 1.0
    elif is_sym:               g = 0.0 if x == y else 1.0
    else:                      g = abs(x - y)
    s += g * g
  return math.sqrt(s / len(sym))


def _centre(members: list[list], sym: list[bool]) -> list:
  "The middle of encoded rows: mean for numbers, most common for symbols; None if unknown."
  out = []
  for j, is_sym in enumerate(sym):
    vs = [m[j] for m in members if m[j] is not None]
    if not vs:       out.append(None)
    elif is_sym:     out.append(Counter(vs).most_common(1)[0][0])
    else:            out.append(sum(vs) / len(vs))
  return out


def families(train_csv: str) -> list[int]:
  "One family label per data row of TRAIN_CSV, in file order."
  tbl  = _load(train_csv)
  rows = tbl.rows
  n    = len(rows)
  if n == 0: return []
  if not tbl.x: return [0] * n
  sym    = [not isinstance(c, ezr.Num) for c in tbl.x]
  E      = _encode(tbl, rows)
  k      = min(_K, n)
  rng    = random.Random(ezr.the.Seed)     # local rng: no global side effects
  cents  = [E[i] for i in rng.sample(range(n), k)]
  labels = None
  for _ in range(_ROUNDS):
    new = [min(range(k), key=lambda j: _dist(e, cents[j], sym)) for e in E]
    if new == labels: break                  # assignments are stable
    labels = new
    for j in range(k):
      members = [e for e, l in zip(E, labels) if l == j]
      if members:                            # keep old centre if empty
        cents[j] = _centre(members, sym)
  return labels


def best(train_csv: str) -> list[int]:
  "Row indices of the unusually good rows (best on all goals together), at most 5% of rows."
  tbl   = _load(train_csv)
  rows  = tbl.rows
  cap   = int(_CUT * len(rows))
  goals = [c for c in tbl.y if isinstance(c, ezr.Num)]
  if cap <= 0 or not goals: return []

  # Per goal: distance from the ideal, smaller is better; unknown is worst.
  G   = [[1.0 if r[c.at] == "?" else abs(ezr.norm(c, r[c.at]) - c.goal)
          for c in goals] for r in rows]
  rms = [math.sqrt(sum(g * g for g in gs) / len(gs)) for gs in G]

  # Lexicographic order puts any dominating row before the rows it dominates,
  # so each row needs checking only against the front found so far.
  order = sorted(range(len(rows)), key=lambda i: (G[i], i))
  front: list[int] = []
  for i in order:
    gi = G[i]
    if not any(all(a <= b for a, b in zip(G[j], gi)) and G[j] != gi
               for j in front):
      front.append(i)

  # Keep the closest to the ideal, capped at 5%; return in file order.
  front.sort(key=lambda i: (rms[i], i))
  return sorted(front[:cap])

