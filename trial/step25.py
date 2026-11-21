# step25.py
"""Blame for the client's two questions: why some customers never respond
to campaigns, and what the complainers have in common.

blame() is step5.blame: input columns ranked by how over-represented their
values are among bad rows. Bad rows are bad on the goals, and the goals
include RESPONSE+ (campaign response) and COMPLAIN- (complaints). The
ranking shows associations in the training data, not proven causes.
"""
import step5


def blame(train_csv) -> list[str]:
  "Input column names, most to blame for bad rows (bad on the goals) first."
  return [str(name) for name in step5.blame(train_csv)]

