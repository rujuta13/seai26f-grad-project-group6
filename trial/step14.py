# step14.py
#!/usr/bin/env python3 -B
"""
step14.py: flow.  Scouting reports arrive one at a time.  stream() keeps a
report if its goals are within the best 5% of the training builds (step4's
keep rule).  drift() warns at the first report where the recent WINDOW goal
values stop looking like the training goal values, by both Cliff's delta
and KS (step4's drift rule).  Each decision uses only train_csv and the
rows that came before it.  Nothing new is needed, so this file delegates to
step4.
"""
from collections.abc import Iterable
import step4

def stream(train_csv: str, rows: Iterable) -> list[bool]:
  "Keep (True) each arriving row whose goals are within the training best 5%."
  return step4.stream(train_csv, rows)

def drift(train_csv: str, rows: Iterable) -> int | None:
  "Index of the first row where recent goal values stop looking like training."
  return step4.drift(train_csv, rows)

