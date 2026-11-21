# step29.py
"""Spot for the client: flag customer records that look wrong, or behave
unlike anyone else. Are they a typo, an impossible value, or just unusual?

odd() is step9.odd, which scores each row by distance from the training
rows and by its goals against the training median and nearest neighbours.
Step9 does not check input values one at a time, so this step adds three
checks:
  * a text cell where a number belongs (a typo such as "3O"): score CAP.
  * an impossible value: AGE outside 0..120, or a negative INCOME: score CAP.
  * an input far from the training median, in robust spreads. Flags (0/1
    columns) and constant columns are skipped, since they are not outliers.
Scores near 1 are typical; several times that is worth checking. Missing
cells ('?') add nothing.
"""
import ezr
import step2
import step9

CAP = step9.CAP
PLAUSIBLE = {"AGE": (0, 120), "INCOME": (0, None)}   # (low, high); None = no limit


def _input_stats(tbl) -> dict:
  "at -> (median, robust spread) for each varying numeric input that is not a flag."
  stats = {}
  for c in tbl.x:
    if type(c) is not ezr.Num: continue
    vals = [r[c.at] for r in tbl.rows if step2._known(r[c.at])]
    if not vals: continue
    if set(vals) <= {0, 1}: continue                 # a flag, not a measure
    med    = step2._pct(vals, .5)
    mad    = 1.4826 * step2._pct([abs(x - med) for x in vals], .5)
    spread = mad or c.sd or 0
    if spread <= 0: continue                         # constant column
    stats[c.at] = (med, spread)
  return stats


def _typo(tbl, full) -> bool:
  "True if a numeric input cell holds text, such as '3O' for '30'."
  return any(type(c) is ezr.Num and full[c.at] != "?"
             and not step2._known(full[c.at]) for c in tbl.x)


def _plausible(v, lo_hi) -> bool:
  "True if V lies within the (low, high) bounds; None means no bound."
  lo, hi = lo_hi
  return not ((lo is not None and v < lo) or (hi is not None and v > hi))


def _input_score(tbl, stats, full) -> float:
  "How odd the numeric inputs of FULL are: impossible values or far outliers."
  s = 0.0
  for c in tbl.x:
    if type(c) is not ezr.Num: continue
    v = full[c.at]
    if not step2._known(v): continue                 # '?' adds nothing
    lo_hi = PLAUSIBLE.get(c.txt.rstrip("+-"))
    if lo_hi and not _plausible(v, lo_hi): return CAP
    if c.at in stats:
      med, sc = stats[c.at]
      s = max(s, abs(v - med) / sc / 3)              # 3 robust spreads out = score 1
  return s


def odd(train_csv, rows) -> list[float]:
  "One suspicion score per row in ROWS (goals included): higher means a likelier glitch, typo, or unlike anything in TRAIN_CSV."
  rows  = list(rows)
  tbl   = step2._load(train_csv)
  stats = _input_stats(tbl)
  fulls = [step2._full(tbl, r) for r in rows]
  # step9 cannot take text in an input cell, so typo rows are scored here.
  clean = [i for i, f in enumerate(fulls) if not _typo(tbl, f)]
  base  = dict(zip(clean, step9.odd(train_csv, [rows[i] for i in clean])))
  out   = []
  for i, f in enumerate(fulls):
    if i not in base:
      out.append(float(CAP))                         # a typo in an input cell
      continue
    s = max(float(base[i]), _input_score(tbl, stats, f))
    out.append(float(min(CAP, s)))
  return out

