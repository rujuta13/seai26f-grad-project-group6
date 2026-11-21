# step22.py
"""Guess a customer's purchases (online and in store) and campaign response.

guess() is step12.guess. The nearest training customers vote: the estimate
is their mean, and low..high is their 10th..90th percentile, so a wide range
means the neighbours disagree. The keys are the goal columns of train_csv,
which include NUMWEBPURCHASES+ (online purchases), NUMSTOREPURCHASES+ (in-store
purchases), and RESPONSE+ (answers the campaign).

Goal cells are never used to make the guess. If a row arrives with its goals
filled in, they are blanked first, so the estimate cannot peek at the answer.
"""
import step2
import step4
import step12


def _inputs_only(train_csv, row):
  "ROW with every goal cell blanked to '?', so only the inputs can inform the guess."
  tbl   = step2._load(train_csv)
  goals = step4._goals(tbl)
  names = {g.txt for g in goals}
  if isinstance(row, dict):
    return {k: v for k, v in row.items() if k not in names}
  if len(row) == len(tbl.names):           # full-width list: blank the goals
    out = list(row)
    for g in goals: out[g.at] = "?"
    return out
  return row                                # input-order list: already inputs only


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
  """Return {goal: (estimate, low, high)} for every goal of TRAIN_CSV.
  ROW has its goal columns removed; any goal cells it carries are ignored.
  'NUMWEBPURCHASES+' is online purchases, 'NUMSTOREPURCHASES+' is in-store
  purchases, and 'RESPONSE+' is whether the customer answers the campaign.
  Low..high is the range the true value is fairly sure to fall in."""
  clean = _inputs_only(train_csv, row)
  out   = step12.guess(train_csv, clean)
  return {goal: (float(est), float(lo), float(hi))
          for goal, (est, lo, hi) in out.items()}

