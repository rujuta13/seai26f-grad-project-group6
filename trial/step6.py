# step6.py
"""Justify the recommended flag set, in terms a skeptic can check.

explain(): the good rows are step1.best (the best 5% on all goals).
It finds the input value most over-represented among them, reports how
often that value occurs in good rows versus all rows, and asks step3.same
whether the goals really differ with and without it. Each claim is a
count or a test anyone can rerun. These are associations in the training
data, not proven causes.

why(): one or two sentences on one build. step4's cutoff says good or
bad. step2.guess's neighbour range shows what the weakest goal was
expected to be.
"""
import ezr
import step1
import step2
import step3
import step4
import step5

TOP = 3    # reasons to report
MIN = 5    # a value must appear in at least this many good rows

say = ezr.say


def _label(col, v, med) -> str:
  "Readable name for one value V of COL; numeric columns were split at MED."
  if type(col) is ezr.Sym: return f"{col.txt} = {say(v)}"
  return f"{col.txt} {'<=' if v else '>'} {say(med)}"


def _best_split(col, rows, gi):
  """The value of COL most over-represented among the good rows GI
  (indexes into ROWS), or None if no value is over-represented."""
  med = 0
  if type(col) is ezr.Num:
    known = [r[col.at] for r in rows if step2._known(r[col.at])]
    med   = step2._pct(known, .5) if known else 0
  vals = step5._values(col, rows, med)
  seen = [v for v in vals if v != "?"]
  good = [vals[i] for i in gi if vals[i] != "?"]
  if not seen or not good: return None
  best = None
  for v in set(seen):
    k = good.count(v)
    if k < MIN: continue
    p_good, p_all = k / len(good), seen.count(v) / len(seen)
    if p_good > p_all and (best is None or p_good - p_all > best[0]):
      best = (p_good - p_all, v, p_good, p_all)
  if best is None: return None
  gain, v, p_good, p_all = best
  return gain, _label(col, v, med), v, vals, p_good, p_all


def _goal_check(tbl, rows, vals, v) -> list[str]:
  "For each goal: do the rows with value V differ from the rest (step3.same)?"
  out = []
  for g in step4._goals(tbl):
    xs = [r[g.at] for r, x in zip(rows, vals)
          if x == v and step2._known(r[g.at])]
    ys = [r[g.at] for r, x in zip(rows, vals)
          if x not in (v, "?") and step2._known(r[g.at])]
    if len(xs) < 2 or len(ys) < 2:
      out.append(f"{g.txt}: too few rows to test")
      continue
    verdict = ("no real difference" if step3.same(xs, ys)
               else "real difference")
    out.append(f"{g.txt}: {verdict} (median {say(step2._pct(xs, .5))} "
               f"with it, {say(step2._pct(ys, .5))} without)")
  return out


def explain(train_csv: str) -> str:
  "Plain reasons, checkable by hand, for what makes rows good."
  tbl  = step2._load(train_csv)
  rows = tbl.rows
  gi   = set(step1.best(train_csv))
  out  = [f"Good rows are the {len(gi)} closest to ideal on all goals "
          f"together (best 5% of {len(rows)}). Each reason below compares "
          f"them with all rows."]
  for g in step4._goals(tbl):
    gv   = [rows[i][g.at] for i in gi if step2._known(rows[i][g.at])]
    allv = [r[g.at] for r in rows if step2._known(r[g.at])]
    way  = "lower is better" if g.goal == 0 else "higher is better"
    out.append(f"{g.txt} ({way}): median {say(step2._pct(gv, .5))} in good "
               f"rows, {say(step2._pct(allv, .5))} overall.")
  found = [h for c in tbl.x if (h := _best_split(c, rows, gi))]
  found.sort(key=lambda h: -h[0])
  if not found:
    out.append("No input value is over-represented among good rows, "
               "so no flag explains goodness here.")
    return "\n".join(out)
  for i, (_, text, v, vals, p_good, p_all) in enumerate(found[:TOP], 1):
    out.append(f"{i}. {text}: in {p_good:.0%} of good rows, "
               f"{p_all:.0%} of all rows.")
    out += ["   " + s for s in _goal_check(tbl, rows, vals, v)]
  out.append("These are associations in this data, not proven causes; "
             "check each before acting.")
  return "\n".join(out)


def why(train_csv: str, row) -> str:
  "One or two sentences: is this build good or bad, and why."
  tbl   = step2._load(train_csv)
  full  = step2._full(tbl, row)
  est   = step2.guess(train_csv, row)
  goals = step4._goals(tbl)
  filled, notes = list(full), []
  for g in goals:
    if not step2._known(full[g.at]):
      est_v, _, _ = est[g.txt]
      filled[g.at] = est_v
      notes.append(f"{g.txt} is unmeasured, so judged at its neighbours' "
                   f"estimate {say(est_v)}")
  ydist   = step4._ydist(tbl, filled)
  cut     = step4._cutoff(train_csv)
  verdict = "good" if ydist <= cut else "bad"
  worst   = max(goals, key=lambda g: abs(ezr.norm(g, filled[g.at]) - g.goal))
  _, lo, hi = est[worst.txt]
  shown   = say(filled[worst.at])
  if not step2._known(full[worst.at]): shown += " (estimated)"
  s1 = (f"This build is {verdict}: its goals sit {say(ydist)} from ideal, "
        f"and good means within {say(cut)}, the best-5% cutoff.")
  s2 = (f"Its weakest goal is {worst.txt} at {shown}, where its nearest "
        f"flag sets measured {say(lo)} to {say(hi)}")
  if notes: s2 += f" ({'; '.join(notes)})"
  return f"{s1} {s2}."

