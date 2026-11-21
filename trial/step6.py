# step6.py
#!/usr/bin/env python3 -B
"""
step6.py: justify.  Plain-words answers for a skeptical engineer.
explain() names the recommended flag set and gives checks anyone can
repeat: goal medians, a significance test, the tree path to the best
leaf, and the blamed inputs.  why() says in one or two sentences why
one build was rated good or bad.
Reuses ezr (tree, ydist, norm, say), step1 (best), step2 (cached table,
probe, guess, quantile), step3 (same), step4 (cut, stream, row), and
step5 (blame).  Only the wording and the path-to-leaf walk are new.
"""
import ezr
import step1, step2, step3, step4, step5

def _test(tbl, node, yes: bool) -> str:
  "One tree split in words: the yes side or the no side."
  name = tbl.names[node.at]
  if type(tbl.cols[node.at]) is ezr.Sym:
    op = "=" if yes else "!="
  else:
    op = "<=" if yes else ">"
  return f"{name} {op} {ezr.say(node.v)}"

def _path(tbl, node, target) -> list[str] | None:
  "The splits from NODE down to TARGET, or None if TARGET is not below NODE."
  if node is target: return []
  for yes, kid in zip((True, False), node.kids):
    sub = _path(tbl, kid, target)
    if sub is not None: return [_test(tbl, node, yes)] + sub
  return None

def explain(train_csv: str) -> str:
  "Plain-words reasons that separate good builds from bad, each one checkable."
  tbl  = step2._tbl(train_csv)
  done = [r for r in tbl.rows if all(r[c.at] != "?" for c in tbl.y)]
  good = [tbl.rows[i] for i in step1.best(train_csv)
          if all(tbl.rows[i][c.at] != "?" for c in tbl.y)]
  if not done or not good:
    return "No training build has every goal known, so there is nothing to explain."
  rec = min(good, key=lambda r: ezr.ydist(tbl, r))
  out = [
    "Recommended flag set (the best of the best 5% on all goals together; "
    f"ydist {ezr.say(ezr.ydist(tbl, rec))}, where 0 is ideal):",
    "  " + ", ".join(f"{c.txt}={ezr.say(rec[c.at])}" for c in tbl.x),
  ]
  for c in tbl.y:   # goal check: the best 5% against every build
    best_xs = sorted(r[c.at] for r in good)
    all_xs  = sorted(r[c.at] for r in done)
    apart   = not step3.same(best_xs, all_xs)
    verdict = ("clearly different by Cliff's delta and KS." if apart
               else "not clearly different by Cliff's delta and KS.")
    out.append(
      f"{c.txt} (better is {'higher' if c.goal else 'lower'}): best-5% median "
      f"{ezr.say(step2._quantile(best_xs, .5))} vs all-builds median "
      f"{ezr.say(step2._quantile(all_xs, .5))}; " + verdict)
  old = ezr.the.Leaf                       # few, big leaves, as in step1
  ezr.the.Leaf = max(old, len(done)//8)
  try:     node = ezr.tree(tbl, done)
  finally: ezr.the.Leaf = old
  leaf = min(ezr.leafs(node), key=lambda z: z.mu)
  path = _path(tbl, node, leaf) or []
  out.append(
    f"A tree on all {len(done)} builds (mean ydist {ezr.say(node.mu)}) finds its "
    f"best leaf (mean ydist {ezr.say(leaf.mu)}, {len(leaf.rows)} builds) at: "
    + (" and ".join(path) if path else "the root, so no single split helped."))
  names = step5.blame(train_csv)[:3]
  out.append("Inputs most tied to bad builds, by eta-squared: "
             + ", ".join(names)
             + ". These are associations in this data, not proven causes.")
  return "\n".join(out)

def why(train_csv: str, row: dict) -> str:
  "One or two sentences: why this build (goals included) is good or bad."
  tbl   = step2._tbl(train_csv)
  probe = step4._row(tbl, row)
  if any(probe[c.at] == "?" for c in tbl.y):   # no goals: rate by neighbours
    est, good_side, notes = step2.guess(train_csv, row), True, []
    for c in tbl.y:
      xs = sorted(r[c.at] for r in tbl.rows if r[c.at] != "?")
      med = step2._quantile(xs, .5)
      e, lo, hi = est[c.txt]
      if (e >= med) != (c.goal == 1): good_side = False
      notes.append(f"{c.txt} about {ezr.say(e)} "
                   f"(range {ezr.say(lo)} to {ezr.say(hi)})")
    verdict = "likely good" if good_side else "likely bad"
    return (f"This build's goals are missing, so it is rated by its nearest "
            f"training builds: {verdict}, which predict {'; '.join(notes)}.")
  y     = ezr.ydist(tbl, probe)
  cut   = step4._cut(train_csv)
  ok    = step4.stream(train_csv, [row])[0]
  verdict = "good" if ok else "bad"
  worst = max(tbl.y, key=lambda c: abs(ezr.norm(c, probe[c.at]) - c.goal))
  top   = step5.blame(train_csv)[0]
  top_v = probe[tbl.names.index(top)]
  return (f"This build is {verdict}: its goals sit {ezr.say(y)} from the best "
          f"possible (the keep cut is {ezr.say(cut)}), and {worst.txt} is the "
          f"furthest off at {ezr.say(probe[worst.at])}. The input most tied to "
          f"bad builds in training is {top}, which this build sets to "
          f"{ezr.say(top_v)}.")

