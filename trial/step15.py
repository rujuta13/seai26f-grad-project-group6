#!/usr/bin/env python3 -B
"step15.py: blame. Which input columns are most to blame for bad rows (bad on the goals)?"
from step5 import blame as _blame

def blame(train_csv) -> list[str]:
  "Input column names, most to blame for bad rows (bad on the goals) first."
  return list(_blame(str(train_csv)))

