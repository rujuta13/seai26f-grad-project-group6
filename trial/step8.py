# step8.py
#!/usr/bin/env python3 -B
"""
step8.py: fix.  Given a bad build (goals removed), find the fewest input
flags to change so that the build becomes good.  "Good" is judged as in
step4.stream: the goals predicted by step2.guess, once placed in the table,
must have ydist within the keep cut from step4._cut.  Columns named in
FROZEN are never changed.  The search tries one changed column, then two,
and so on.  The first size with a good answer wins, and among answers of
that size the one with the lowest predicted ydist is returned.  The search
stops after MAX_EVALS guesses.  If no good fix is found by then, the result
is the least-bad change tried.
"""
from itertools import combinations, product
import ezr
import step2, step4

MAX_EVALS = 1000   # cap on guesses, so the search stays bounded

def _values(col, tbl) -> list:
  "Distinct known values of input COL in the training rows, thinned to about 10."
  vs = {r[col.at] for r in tbl.rows if r[col.at] != "?"}
  vs = sorted(vs, key=(lambda v: v) if type(col) is ezr.Num else str)
  if len(vs) > 10:
    vs = [vs[round(i*(len(vs)-1)/9)] for i in range(10)]
  return vs

def fix(train_csv: str, row: dict, frozen=()) -> dict:
  "Same row's inputs, with as few values changed as possible so the build is good."
  tbl    = step2._tbl(train_csv)
  cut    = step4._cut(train_csv)
  frozen = set(frozen)
  base   = {}
  for c in tbl.x:
    v = row.get(c.txt, "?")
    base[c.txt] = "?" if v is None else v
  cache  = {}        # input values -> (good?, ydist), so repeats cost nothing
  evals  = [0]

  def judge(changes: dict) -> tuple[bool, float]:
    "Is the build with CHANGES applied good, judged on its predicted goals?"
    cand = {**base, **changes}
    key  = tuple(cand[c.txt] for c in tbl.x)
    if key not in cache:
      evals[0] += 1
      est   = step2.guess(train_csv, cand)
      probe = step2._probe(tbl, cand)
      for c in tbl.y: probe[c.at] = est[c.txt][0]   # predicted goals
      y = ezr.ydist(tbl, probe)
      cache[key] = (y <= cut, y)
    return cache[key]

  ok, y = judge({})
  if ok: return dict(row)                 # already good: nothing to change

  cols    = [c for c in tbl.x if c.txt not in frozen]
  choices = {c.txt: [v for v in _values(c, tbl) if v != base[c.txt]]
             for c in cols}
  names   = [n for n in choices if choices[n]]

  def trials():
    "Every change set, smallest first: one column, then two, and so on."
    for k in range(1, len(names)+1):
      for combo in combinations(names, k):
        for vals in product(*(choices[n] for n in combo)):
          yield k, dict(zip(combo, vals))

  best_k, found = None, None
  best_any      = (y, {})                 # least-bad change, if nothing is good
  for k, changes in trials():
    if best_k is not None and k > best_k: break   # finished the smallest good size
    if evals[0] >= MAX_EVALS: break
    good, yy = judge(changes)
    if yy < best_any[0]: best_any = (yy, changes)
    if good and (found is None or yy < found[0]):
      found, best_k = (yy, changes), k
  pick = found or best_any
  return {**row, **pick[1]}

