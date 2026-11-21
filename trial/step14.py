# step14.py
"""
step14: flow, for the client.  Scouting reports arrive one at a time.
stream() flags the ones worth following up (good on the goals).  drift()
warns when this season's players stop looking like last season's.
Both reuse step4 unchanged.  Each decision uses train_csv and the rows
seen so far, never a later row.
"""
import step4


def stream(train_csv, rows) -> list[bool]:
  "Keep (True) each row whose goals are as good as the best training rows."
  if not step4._train_ys(train_csv):
    raise ValueError("stream: train_csv has no rows to compare against")
  return step4.stream(train_csv, rows)


def drift(train_csv, rows) -> int | None:
  "Index of the first row where the recent rows stop matching training."
  return step4.drift(train_csv, rows)

