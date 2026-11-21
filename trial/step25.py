# step25.py
"""
step25: blame, for the client.  Two asks: why do some customers never
respond to campaigns (RESPONSE+), and what do the customers who complain
have in common (COMPLAIN-)?  So only those two goals count toward "bad".
The other goals are renamed to end in X, which makes ezr skip them (the
same trick step7 uses).  Then step5's blame runs unchanged on that copy.
"""
import os
import tempfile
import step5

FOCUS = ("RESPONSE+", "COMPLAIN-")   # the two goals the client asks about


def _lines(train_csv: str) -> list[str]:
  "The non-blank lines of TRAIN_CSV, read the way ezr.csv reads them."
  with open(os.path.expanduser(train_csv), encoding="utf-8-sig") as f:
    return [line for line in f if line.strip()]


def _names(lines: list[str]) -> list[str]:
  "Header names of the csv, stripped of spaces and line ends."
  return [s.strip() for s in lines[0].split(",")]


def _focus_csv(train_csv: str) -> str:
  "A temp csv like TRAIN_CSV in which only FOCUS goals still count. Returns its path."
  lines = _lines(train_csv)
  names = [s if (not s or s in FOCUS or s[-1] not in "+-!") else s[:-1] + "X"
           for s in _names(lines)]
  fd, path = tempfile.mkstemp(suffix=".csv")
  with os.fdopen(fd, "w", encoding="utf-8") as f:
    f.write(",".join(names) + "\n")
    f.writelines(lines[1:])
  return path


def blame(train_csv: str) -> list[str]:
  """Input column names, most to blame for bad rows (bad on the goals)
  first. Bad rows here are bad on RESPONSE+ and COMPLAIN- together.
  If neither of those goals is in the file, every goal counts instead.
  Ties keep the order of the header."""
  names = _names(_lines(train_csv))
  if not any(s in FOCUS for s in names):
    return list(step5.blame(train_csv))     # no such goals: use all of them
  path = _focus_csv(train_csv)
  try:
    return list(step5.blame(path))
  finally:
    os.remove(path)

