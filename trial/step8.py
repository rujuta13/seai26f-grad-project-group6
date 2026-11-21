# step8.py
"""Fix: flip as few input flags as possible to make a bad build good.

fix() searches over changes to the input (x) columns. A candidate is judged
the way step6.why judges a build: its goals are estimated by step2.guess
(the nearest training rows vote), and it is good when its distance from
ideal is within step4's best-5% cutoff. Changes are tried exhaustively in
size order (1, then 2, then 3 flags), so the answer uses the fewest flips
up to that size. Beyond that, a greedy walk is used. Columns named in
frozen are never changed.
"""
import itertools

import ezr
import step2
import step4

MAX_EXACT = 3      # exhaustive search up to this many changed columns


def _judge(train_csv, tbl, cur: dict) -> float:
  "Estimated distance from ideal for input dict CUR, goals guessed from neighbours."
  est  = step2.guess(train_csv, cur)
  full = step2._full(tbl, cur)
  for g in step4._goals(tbl):
    full[g.at] = est[g.txt][0]
  return step4._ydist(tbl, full)


def fix(train_csv, row, frozen=()) -> dict:
  "Inputs of ROW with as few values changed as possible so it is good; FROZEN never change."
  if isinstance(frozen, str): frozen = (frozen,)
  tbl   = step2._load(train_csv)
  full  = step2._full(tbl, row)
  names = {c.txt for c in tbl.x}
  unknown = [f for f in frozen if f not in names]
  if unknown:
    raise ValueError(f"fix: frozen columns are not inputs: {unknown}; "
                     f"have {sorted(names)}")

  cur  = {c.txt: full[c.at] for c in tbl.x}          # inputs of the row
  cut  = step4._cutoff(train_csv)
  free = [c for c in tbl.x if c.txt not in frozen]
  vals = {c.txt: sorted({r[c.at] for r in tbl.rows if r[c.at] != "?"},
                        key=str)
          for c in free}

  here = _judge(train_csv, tbl, cur)
  if here <= cut: return cur                          # already good

  # exhaustive, by number of changed columns: the first size that works wins
  for k in range(1, min(MAX_EXACT, len(free)) + 1):
    found = []
    for cols in itertools.combinations(free, k):
      options = [[(c.txt, v) for v in vals[c.txt] if v != cur[c.txt]]
                 for c in cols]
      for picks in itertools.product(*options):
        cand = dict(cur)
        for name, v in picks: cand[name] = v
        d = _judge(train_csv, tbl, cand)
        if d <= cut: found.append((d, cand))
    if found:
      return min(found, key=lambda t: t[0])[1]

  # greedy fallback: take the single change that helps most, one column at a time
  d, cand, changed = here, dict(cur), set()
  while d > cut:
    moves = [(c.txt, v) for c in free if c.txt not in changed
             for v in vals[c.txt] if v != cand[c.txt]]
    if not moves: break
    scored = []
    for name, v in moves:
      trial = dict(cand)
      trial[name] = v
      scored.append((_judge(train_csv, tbl, trial), name, trial))
    nd, name, trial = min(scored, key=lambda t: t[0])
    if nd >= d: break
    cand, d = trial, nd
    changed.add(name)
  if d <= cut: return cand
  raise ValueError("fix: no good build reachable by changing the unfrozen flags")

