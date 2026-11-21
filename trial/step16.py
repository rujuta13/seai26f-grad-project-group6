# step16.py
#!/usr/bin/env python3 -B
"""
step16.py: justify, for the manager and the scout.
explain() gives reasons to sign a player: goal medians, a significance
test, the tree path to the best group, and the inputs most tied to weak
players.  why() says in one or two sentences why one player was rated
good or bad, and when goals are missing, what the nearest players predict.
Reuses step1 (best), step2 (cached table, guess, quantile), step3 (same),
step4 (cut, stream, row), step6 (tree path helper), and step15 (blame,
adjusted so identifier-like columns do not look guilty).  Only the player
wording is new.
"""
import ezr
import step1, step2, step3, step4, step6, step15

def explain(train_csv: str) -> str:
  "Reasons, in plain words, that separate strong players from weak ones, each one checkable."
  tbl  = step2._tbl(train_csv)
  done = [r for r in tbl.rows if all(r[c.at] != "?" for c in tbl.y)]
  good = [tbl.rows[i] for i in step1.best(train_csv)
          if all(tbl.rows[i][c.at] != "?" for c in tbl.y)]
  if not done or not good:
    return "No player has every goal known, so there is nothing to explain."
  rec   = min(good, key=lambda r: ezr.ydist(tbl, r))
  names = step15.blame(train_csv)
  out = [
    "Profile of the strongest player found (best 5% on all goals together; "
    f"ydist {ezr.say(ezr.ydist(tbl, rec))}, where 0 is ideal):",
    "  " + ", ".join(f"{n}={ezr.say(rec[tbl.names.index(n)])}"
                     for n in names[:3]),
  ]
  for c in tbl.y:   # goal check: the strong 5% against every player
    best_xs = sorted(r[c.at] for r in good)
    all_xs  = sorted(r[c.at] for r in done)
    apart   = not step3.same(best_xs, all_xs)
    verdict = ("clearly different by Cliff's delta and KS, so the gap is probably real."
               if apart else
               "not clearly different by Cliff's delta and KS, so the gap may be noise.")
    out.append(
      f"{c.txt} (better is {'higher' if c.goal else 'lower'}): strong-5% median "
      f"{ezr.say(step2._quantile(best_xs, .5))} vs all-players median "
      f"{ezr.say(step2._quantile(all_xs, .5))}; " + verdict)
  old = ezr.the.Leaf                       # few, big groups, as in step1
  ezr.the.Leaf = max(old, len(done)//8)
  try:     node = ezr.tree(tbl, done)
  finally: ezr.the.Leaf = old
  leaf = min(ezr.leafs(node), key=lambda z: z.mu)
  path = step6._path(tbl, node, leaf) or []
  out.append(
    f"A tree on all {len(done)} players (mean ydist {ezr.say(node.mu)}) finds its "
    f"best group (mean ydist {ezr.say(leaf.mu)}, {len(leaf.rows)} players) at: "
    + (" and ".join(path) if path else "the root, so no single split helped."))
  out.append("Inputs most tied to weak players, after discounting identifier-like "
             "columns: " + ", ".join(names[:3])
             + ". These are associations in this data; confirm them on video "
             "before relying on them.")
  return "\n".join(out)

def why(train_csv: str, row: dict) -> str:
  "One or two sentences: why this player (goals included) is rated good or bad."
  tbl   = step2._tbl(train_csv)
  probe = step4._row(tbl, row)
  if any(probe[c.at] == "?" for c in tbl.y):   # no goals: rate by nearest players
    est, good_side, notes = step2.guess(train_csv, row), True, []
    for c in tbl.y:
      xs = sorted(r[c.at] for r in tbl.rows if r[c.at] != "?")
      med = step2._quantile(xs, .5)
      e, lo, hi = est[c.txt]
      if (e >= med) != (c.goal == 1): good_side = False
      notes.append(f"{c.txt} about {ezr.say(e)} "
                   f"(range {ezr.say(lo)} to {ezr.say(hi)})")
    verdict = "likely good" if good_side else "likely bad"
    return (f"This player's goals are missing, so he is rated by his nearest "
            f"training players: {verdict}, who predict {'; '.join(notes)}.")
  y       = ezr.ydist(tbl, probe)
  cut     = step4._cut(train_csv)
  ok      = step4.stream(train_csv, [row])[0]
  verdict = "good" if ok else "bad"
  worst   = max(tbl.y, key=lambda c: abs(ezr.norm(c, probe[c.at]) - c.goal))
  top     = step15.blame(train_csv)[0]
  top_v   = probe[tbl.names.index(top)]
  return (f"This player is rated {verdict}: his goals sit {ezr.say(y)} from the best "
          f"possible (the keep cut is {ezr.say(cut)}), and {worst.txt} is the "
          f"furthest off at {ezr.say(probe[worst.at])}, so check that on video. "
          f"The input most tied to weak players in training is {top}, which "
          f"this player has at {ezr.say(top_v)}.")

