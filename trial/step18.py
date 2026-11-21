# step18.py
"""Fix: the fewest input changes that make a bad row good, with frozen columns kept.

fix() is step8.fix. It searches only the input (x) columns, judges each
candidate with step2.guess and step4's best-5% cutoff, and never changes a
column named in frozen. This step is the client-facing entry point, and it
checks that no frozen column moved.
"""
import step2
import step8


def fix(train_csv, row, frozen=()) -> dict:
  "Inputs of ROW with as few values changed as possible so it is good; FROZEN never change."
  if isinstance(frozen, str): frozen = (frozen,)
  frozen = tuple(frozen)
  out    = step8.fix(train_csv, row, frozen=frozen)
  tbl    = step2._load(train_csv)
  full   = step2._full(tbl, row)
  before = {c.txt: full[c.at] for c in tbl.x}
  moved  = [f for f in frozen if out.get(f) != before.get(f)]
  if moved:
    raise ValueError(f"fix: frozen columns changed: {moved}")
  return out

