#!/usr/bin/env python3 -B
"step25.py: blame. Why do some customers never respond to campaigns, and what do the customers who complain have in common?"
from step5 import blame as _blame

def blame(train_csv) -> list[str]:
  "Input column names, most to blame for bad rows (bad on the goals) first."
  # Goals such as RESPONSE+ and COMPLAIN- define "bad" together with the other goals.
  return list(_blame(str(train_csv)))

