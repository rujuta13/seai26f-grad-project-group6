# step16.py
"""
step16: justify, for the client.  Two asks: why should the manager sign
this player, and why was this player rated low, with reasons that can be
checked on video.  step6 already answers both in plain words from ezr's
tree, so this file only names them for the client.
"""
import step6


def explain(train_csv) -> str:
  "Reasons, in plain words a skeptical expert can check, for what makes rows good."
  return step6.explain(train_csv)


def why(train_csv, row) -> str:
  "One or two sentences on why this row (goals included) is good or bad."
  return step6.why(train_csv, row)

