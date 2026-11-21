# step26.py
"""
step26: justify, for the client.  Two asks: why should the sales director
send the next campaign to this group, and why was this customer marked low
value, with reasons the team can act on.  step6 already answers both in
plain words from ezr's tree, so this file only names them for the client.
"""
import step6


def explain(train_csv) -> str:
  "Reasons, in plain words a skeptical expert can check, for what makes rows good."
  return step6.explain(train_csv)


def why(train_csv, row) -> str:
  "One or two sentences on why this row (goals included) is good or bad."
  return step6.why(train_csv, row)

