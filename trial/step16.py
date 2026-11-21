#!/usr/bin/env python3 -B
"step16.py: justify. Plain-word reasons a manager can check on video, and why one player is rated good or bad."
from step6 import explain as _explain, why as _why

def explain(train_csv) -> str:
  "Reasons, in plain words a skeptical expert can check, for what makes rows good."
  return _explain(str(train_csv))

def why(train_csv, row) -> str:
  "One or two sentences on why this row (goals included) is good or bad."
  return _why(str(train_csv), row)

