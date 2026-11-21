# step15.py
"""Blame: which input columns explain the bad rows, for the client's questions.

Bad rows are those bad on the goals (step5's definition). The ranking
comes from step5.blame, which reuses step4's goal distance and step2's
neighbour helpers. Each input column is scored by how much more often one
of its values appears among bad rows than among all rows.
"""
import step5


def blame(train_csv) -> list[str]:
  "Input column names, most to blame for bad rows (bad on the goals) first."
  return step5.blame(train_csv)

