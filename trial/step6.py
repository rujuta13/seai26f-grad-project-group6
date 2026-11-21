# step6.py
import statistics
import ezr
import step2
import step3
import step4
import step5

def _split(tbl, score) -> tuple[list, list]:
  "Good rows beat the train average on the goals; bad rows do worse."
  ws   = [score(r) for r in tbl.rows]
  good = [r for r, w in zip(tbl.rows, ws) if w > 0]
  bad  = [r for r, w in zip(tbl.rows, ws) if w < 0]
  return good, bad

def _contrast(c, gv: list, bv: list) -> str:
  "One checkable sentence: how input C differs between good and bad rows."
  if not gv or not bv:
    return f"{c.txt}: missing in one group, so no comparison."
  if type(c) is ezr.Sym:
    vals = set(gv) | set(bv)
    fg = lambda v: gv.count(v) / len(gv)
    fb = lambda v: bv.count(v) / len(bv)
    v  = max(vals, key=lambda v: abs(fg(v) - fb(v)))
    return (f"{c.txt}={v} is in {100*fg(v):.0f}% of good builds "
            f"and {100*fb(v):.0f}% of bad ones.")
  mg, mb = statistics.median(gv), statistics.median(bv)
  real   = not step3.same(gv, bv)      # same() True means noise
  return (f"{c.txt}: median {mg:.2f} in good builds vs {mb:.2f} in bad ones "
          + ("(a real difference)." if real else "(within noise)."))

def explain(train_csv: str) -> str:
  "Plain reasons, checkable by hand, for what makes rows good."
  tbl, score = step4._scorer(train_csv)
  good, bad  = _split(tbl, score)
  out = [f"{len(good)} of {len(tbl.rows)} builds are good: closer to the "
         "ideal on the goals than the average build."]
  if not good or not bad:
    out.append("Good and bad rows do not both exist, so no contrast is possible.")
    return "\n".join(out)

  # goals: a sanity check that the good/bad split means what it says
  for c in tbl.y:
    gv = [r[c.at] for r in good if r[c.at] != "?"]
    bv = [r[c.at] for r in bad  if r[c.at] != "?"]
    if gv and bv:
      want = "lower" if c.goal == 0 else "higher"
      out.append(f"{c.txt}: good builds average {statistics.mean(gv):.2f}, "
                 f"bad builds average {statistics.mean(bv):.2f} "
                 f"({want} is better).")

  # inputs: the three columns most to blame, with the evidence for each
  by_name = {tbl.names[c.at]: c for c in tbl.x}
  for name in step5.blame(train_csv)[:3]:
    c  = by_name[name]
    gv = [r[c.at] for r in good if r[c.at] != "?"]
    bv = [r[c.at] for r in bad  if r[c.at] != "?"]
    out.append("- " + _contrast(c, gv, bv))

  # a tree on the inputs: the first split it chooses, and what each side scores
  root = ezr.tree(tbl, tbl.rows)
  if root.kids:
    yes, no = root.kids
    op = "=" if type(tbl.cols[root.at]) is ezr.Sym else "<="
    out.append(f"A tree on the inputs first splits on "
               f"{tbl.names[root.at]} {op} {ezr.say(root.v)}: rows where that "
               f"holds have mean ydist {100*yes.mu:.0f} (0 is ideal), the rest "
               f"{100*no.mu:.0f}.")
  return "\n".join(out)

def why(train_csv: str, row) -> str:
  "One or two sentences: is ROW (goals included) good or bad, and why."
  tbl, score = step4._scorer(train_csv)
  full    = step2._row(tbl, row)
  missing = [c.txt for c in tbl.y if full[c.at] == "?"]
  if missing:
    return (f"This row cannot be rated: its goal value(s) "
            f"{', '.join(missing)} are missing.")
  s       = score(full)
  verdict = "good" if s > 0 else "bad" if s < 0 else "average"
  parts = []
  for c in tbl.y:
    v      = full[c.at]
    better = v < c.mu if c.goal == 0 else v > c.mu
    parts.append(f"{c.txt} is {ezr.say(v)}, "
                 f"{'better' if better else 'worse'} than the train average "
                 f"of {ezr.say(c.mu)}")
  return (f"This row is {verdict} (score {s:.0f}, where 0 is an average build "
          f"and 100 the best): " + "; and ".join(parts) + ".")

