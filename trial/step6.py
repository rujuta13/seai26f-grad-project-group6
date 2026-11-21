#!/usr/bin/env python3 -B
"step6.py: justify. Why is a flag set good or bad, in words a skeptic can check?"
from ezr import Sym, candidates, norm, say, ydist
from step1 import TOP, best
from step2 import _load
from step4 import _ref, _row
from step5 import blame

TOP_SPLITS  = 3   # most-to-blame input columns to explain
MIN_SUPPORT = 3   # a split must cover at least this many builds

def explain(train_csv) -> str:
  "Plain-word reasons, each a split you can count by hand, for what makes builds good."
  tbl  = _load(str(train_csv))
  rows = tbl.rows
  n    = len(rows)
  if n == 0: return "There are no builds to explain."
  good = set(best(str(train_csv)))
  if not good: return "No build is good enough to learn from."
  base  = len(good) / n
  goals = "; ".join(f"{c.txt} ({'higher' if c.goal else 'lower'} is better)"
                    for c in tbl.y)
  out = [f"A build is good when it is among the best {100*TOP:.0f}% "
         f"on the goals together ({goals}).",
         f"{len(good)} of {n} builds ({100*base:.0f}%) meet that bar."]
  for name in blame(str(train_csv))[:TOP_SPLITS]:
    col   = tbl.cols[tbl.names.index(name)]
    found = None                       # (rate, hits, support, text)
    for v, go in candidates(col, rows):
      yes = [i for i, r in enumerate(rows) if go(r)]
      if len(yes) < MIN_SUPPORT or len(yes) == n: continue
      hits = sum(1 for i in yes if i in good)
      rate = hits / len(yes)
      cond = (f"{name} = {say(v)}" if type(col) is Sym
              else f"{name} <= {say(v)}")
      if found is None or (rate, len(yes)) > found[:2]:
        found = (rate, hits, len(yes), cond)
    if found and found[0] > base:
      rate, hits, k, cond = found
      out.append(f"{cond}: {hits} of {k} such builds ({100*rate:.0f}%) "
                 f"are good, against {100*base:.0f}% overall.")
  if len(out) == 2:
    out.append("No single input split makes good builds clearly more likely, "
               "so the goals are the only firm evidence.")
  return "\n".join(out)

def why(train_csv, row) -> str:
  "One or two sentences: is this build good or bad, and which goal drives it?"
  tbl, cutoff = _ref(str(train_csv))
  full        = _row(tbl, row)
  missing     = [c.txt for c in tbl.y if full[c.at] == "?"]
  if missing:
    return f"Cannot judge this build: its goal(s) {', '.join(missing)} are unknown."
  d       = ydist(tbl, full)
  verdict = "good" if d <= cutoff else "bad"
  worst   = max(tbl.y, key=lambda c: abs(norm(c, full[c.at]) - c.goal))
  want    = "higher" if worst.goal else "lower"
  return (f"This build is {verdict}: its distance from the ideal goals is "
          f"{say(d)}, against a cutoff of {say(cutoff)} for the best builds. "
          f"Its weakest goal is {worst.txt} at {say(full[worst.at])}, "
          f"where {want} is better.")

