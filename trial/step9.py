# step9.py
"""Odd rows: benchmark results that look like glitches, or flag sets unlike any seen.

odd() gives each row a suspicion score (higher = more suspicious) from three checks against TRAIN_CSV:
  * distance: how far the row's flags sit from its nearest training row, compared
    with how far training rows sit from their own nearest neighbour (step4's
    leave-one-out reference). Far beyond that means unlike anything seen.
  * goal range: a goal far from the training median, in robust spreads (median and
    MAD), or a goal cell that is text rather than a number (a likely typo).
  * goal neighbours: a goal far from what the nearest flag sets measured
    (step2.guess's neighbour band).
A score near 1 is the typical edge of the data; several times that is odd.
Missing cells ('?') are not evidence, so they add nothing.
"""
import ezr
import step2
import step4

CAP = 1e3     # scores are capped here, so one bad cell cannot swamp the rest


def odd(train_csv: str, rows: list) -> list[float]:
  "One suspicion score per row of ROWS: higher means a likelier glitch, or unlike TRAIN_CSV."
  tbl   = step2._load(train_csv)
  ref   = step4._reference(train_csv)
  goals = step4._goals(tbl)
  nx    = max(1, len(tbl.x))
  far   = max(step2._pct(ref, .9), nx ** -.5)  # one flipped flag = 1 unit
  stats = {}                                    # goal -> (median, robust spread)
  for g in goals:
    vals = [r[g.at] for r in tbl.rows if step2._known(r[g.at])]
    if not vals: continue
    med  = step2._pct(vals, .5)
    mad  = 1.4826 * step2._pct([abs(x - med) for x in vals], .5)
    stats[g.txt] = (med, mad or g.sd or 1e-9)
  out = []
  for row in rows:
    full = step2._full(tbl, row)
    est  = step2.guess(train_csv, row)
    s    = min(CAP, step4._nearness(tbl, full) / far)
    for g in goals:
      v = full[g.at]
      if v == "?": continue
      if not step2._known(v):                 # text where a number belongs: a typo
        s = CAP
        continue
      if g.txt in stats:
        med, sc = stats[g.txt]
        s = max(s, abs(v - med) / sc / 3)     # 3 robust spreads out = score 1
      if g.txt in est:
        e, lo, hi = est[g.txt]
        s = max(s, abs(v - e) / max(hi - lo, 1e-9))
    out.append(float(min(CAP, s)))
  return out

