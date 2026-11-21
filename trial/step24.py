# step24.py
#!/usr/bin/env python3 -B
"""
step24.py: flow.  New customers sign up one at a time.  stream() keeps a
sign-up if its goals are within the best 5% of the training customers, so
those are the ones worth a welcome offer.  drift() warns at the first
sign-up where the recent goal values stop looking like the training
customers.  Each decision uses only train_csv and the rows before it.
Nothing new is needed, so this file delegates to step14, which delegates
to step4.
"""
from collections.abc import Iterable
import step14

def stream(train_csv, rows: Iterable) -> list[bool]:
  "Keep (True) each arriving sign-up whose goals are within the training best 5%."
  return step14.stream(train_csv, rows)

def drift(train_csv, rows: Iterable) -> int | None:
  "Index of the first sign-up where recent goal values stop looking like training, or None."
  return step14.drift(train_csv, rows)

