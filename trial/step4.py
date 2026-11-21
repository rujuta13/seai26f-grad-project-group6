# step4.py
import random
import ezr
import step2
import step3

W      = 20    # drift: how many recent rows to compare, at once
SAMPLE = 100   # drift: how many train rows to measure the reference with

_scorers = {}  # train_csv -> (TBL, wins scorer), built once
_refs    = {}  # train_csv -> (TBL, leave-one-out gaps of sampled train rows)

def _scorer(train_csv: str):
  "A wins scorer over the train table, built once per train_csv."
  if train_csv not in _scorers:
    tbl = step2._table(train_csv)
    if not tbl.rows: raise ValueError(f"no rows in {train_csv}")
    _scorers[train_csv] = (tbl, ezr.wins(tbl))
  return _scorers[train_csv]

def _complete(tbl, row):
  "ROW in full column layout, or None if a goal is missing (can't judge it)."
  full = step2._row(tbl, row)
  if any(full[c.at] == "?" for c in tbl.y): return None
  return full

def stream(train_csv, rows) -> list[bool]:
  "For each row, in order: True if its goals beat the train average."
  tbl, score = _scorer(train_csv)
  out = []
  for row in rows:
    full = _complete(tbl, row)
    out.append(full is not None and score(full) > 0)
  return out

def _reference(train_csv: str):
  "Leave-one-out nearest-neighbour gaps of sampled train rows: what 'normal' looks like."
  if train_csv not in _refs:
    tbl  = step2._table(train_csv)
    rows = tbl.rows
    if len(rows) < 2: raise ValueError(f"need at least 2 rows in {train_csv}")
    rng   = random.Random(ezr.the.Seed)          # reproducible sample
    picks = rng.sample(range(len(rows)), min(SAMPLE, len(rows)))
    gaps  = [min(ezr.xdist(tbl, rows[i], s)
                 for j, s in enumerate(rows) if j != i)
             for i in picks]
    _refs[train_csv] = (tbl, gaps)
  return _refs[train_csv]

def drift(train_csv, rows) -> int | None:
  "Index of the first row where the last W new rows stop looking like train_csv."
  tbl, ref = _reference(train_csv)
  near = []                                      # each new row's nearest-train gap
  for i, row in enumerate(rows):
    full = step2._row(tbl, row)
    near.append(min(ezr.xdist(tbl, full, t) for t in tbl.rows))
    if len(near) >= W and not step3.same(ref, near[-W:]):
      return i
  return None

