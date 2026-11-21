# step26.py
#!/usr/bin/env python3 -B
"""
step26.py: justify, for the sales director and the customer team.
explain() gives reasons to target a group with the next campaign: the
profile of the best customer found, goal medians against all customers,
a significance test, the campaign response rate, the tree path to the
best group, and the inputs most tied to low-value customers.  why() says
in one or two sentences why one customer is rated high-value or
low-value, and when goals are missing, what the nearest customers predict.
Reuses step1 (best), step2 (cached table, guess, quantile), step3 (same),
step4 (cut, stream, row), step6 (tree path helper), and step15 (blame,
adjusted so identifier-like columns do not look guilty).  Only the
customer wording and the campaign-response check are new.
"""
import ezr
import step1, step2, step3, step4, step6, step15

def explain(train_csv: str) -> str:
  "Reasons, in plain words, that a sales director can check, for targeting the best group."
  tbl  = step2._tbl(train_csv)
  done = [r for r in tbl.rows if all(r[c.at] != "?" for c in tbl.y)]
  good = [tbl.rows[i] for i in step1.best(train_csv)
          if all(tbl.rows[i][c.at] != "?" for c in tbl.y)]
  if not done or not good:
    return "No customer has every goal known, so there is nothing to explain."
  rec   = min(good, key=lambda r: ezr.ydist(tbl, r))
  names = step15.blame(train_csv)
  out = [
    "Profile of the strongest customer found (best 5% on all goals together; "
    f"ydist {ezr.say(ezr.ydist(tbl, rec))}, where 0 is ideal):",
    "  " + ", ".join(f"{n}={ezr.say(rec[tbl.names.index(n)])}"
                     for n in names[:3]),
  ]
  for c in tbl.y:   # goal check: the high-value 5% against every customer
    best_xs = sorted(r[c.at] for r in good)
    all_xs  = sorted(r[c.at] for r in done)
    apart   = not step3.same(best_xs, all_xs)
    verdict = ("clearly different by Cliff's delta and KS, so the gap is probably real."
               if apart else
               "not clearly different by Cliff's delta and KS, so the gap may be noise.")
    out.append(
      f"{c.txt} (better is {'higher' if c.goal else 'lower'}): high-value median "
      f"{ezr.say(step2._quantile(best_xs, .5))} vs all-customers median "
      f"{ezr.say(step2._quantile(all_xs, .5))}; " + verdict)
  for c in tbl.y:   # campaign check: the response goal, as a rate
    if c.txt != "RESPONSE+": continue
    hit  = sum(r[c.at] for r in good) / len(good)
    base = sum(r[c.at] for r in done) / len(done)
    cmp  = "above" if hit > base else "not above"
    out.append(
      f"Campaign response: {ezr.say(100*hit)}% of the high-value group responded, "
      f"against {ezr.say(100*base)}% of all customers, which is {cmp} the average, so "
      + ("this is a reason to target them." if hit > base
         else "this is not a reason to target them."))
  old = ezr.the.Leaf                       # few, big groups, as in step1
  ezr.the.Leaf = max(old, len(done)//8)
  try:     node = ezr.tree(tbl, done)
  finally: ezr.the.Leaf = old
  leaf = min(ezr.leafs(node), key=lambda z: z.mu)
  path = step6._path(tbl, node, leaf) or []
  out.append(
    f"A tree on all {len(done)} customers (mean ydist {ezr.say(node.mu)}) finds its "
    f"best group (mean ydist {ezr.say(leaf.mu)}, {len(leaf.rows)} customers) at: "
    + (" and ".join(path) if path else "the root, so no single split helped."))
  out.append("Inputs most tied to low-value customers, after discounting identifier-like "
             "columns: " + ", ".join(names[:3])
             + ". These are associations in this data; test them with a small "
             "pilot campaign before relying on them.")
  return "\n".join(out)

def why(train_csv: str, row: dict) -> str:
  "One or two sentences: why this customer (goals included) is high-value or low-value."
  tbl   = step2._tbl(train_csv)
  probe = step4._row(tbl, row)
  if any(probe[c.at] == "?" for c in tbl.y):   # no goals: rate by nearest customers
    est, good_side, notes = step2.guess(train_csv, row), True, []
    for c in tbl.y:
      xs  = sorted(r[c.at] for r in tbl.rows if r[c.at] != "?")
      med = step2._quantile(xs, .5)
      e, lo, hi = est[c.txt]
      if (e >= med) != (c.goal == 1): good_side = False
      notes.append(f"{c.txt} about {ezr.say(e)} "
                   f"(range {ezr.say(lo)} to {ezr.say(hi)})")
    verdict = "likely high-value" if good_side else "likely low-value"
    return (f"This customer's goals are missing, so they are rated by the nearest "
            f"training customers: {verdict}, who predict {'; '.join(notes)}.")
  y       = ezr.ydist(tbl, probe)
  cut     = step4._cut(train_csv)
  ok      = step4.stream(train_csv, [row])[0]
  verdict = "high-value (good)" if ok else "low-value (bad)"
  worst   = max(tbl.y, key=lambda c: abs(ezr.norm(c, probe[c.at]) - c.goal))
  top     = step15.blame(train_csv)[0]
  top_v   = probe[tbl.names.index(top)]
  return (f"This customer is rated {verdict}: their goals sit {ezr.say(y)} from the best "
          f"possible (the keep cut is {ezr.say(cut)}), and {worst.txt} is the "
          f"furthest off at {ezr.say(probe[worst.at])}. The input most tied to "
          f"low-value customers in training is {top}, which this customer has at "
          f"{ezr.say(top_v)}.")

