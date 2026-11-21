# step4.py
"""Stream flow: keep good nightly rows as they arrive, and warn on drift.

stream(): a new row is kept when its goals fall inside the best 5% of
TRAIN_CSV (the same cut step1.best makes). It uses only TRAIN_CSV and
the row itself.

drift(): nearness is how close a new row sits to its nearest training
row over the x columns. Training rows give their own reference nearness
(leave-one-out). New rows are tested in non-overlapping blocks of WINDOW,
and the first block that no longer matches the reference raises the
warning. The test is KS at a stricter level than step3.same's 95%, since
the blocks are checked repeatedly and a looser bar would false-alarm.
"""
import ezr
import step1
import step2

WINDOW = 25        # rows per drift block
REF    = 100       # training rows sampled as the nearness reference
ALPHA  = 1.95      # KS at about 99.9%: repeated looks need a stricter bar

_cut: dict = {}    # train_csv -> ydist of the best-5% cutoff
_ref: dict = {}    # train_csv -> sorted nearness of sampled training rows


def _goals(tbl) -> list:
  "The goal columns (+ or -), skipping any klass column."
  return [g for g in tbl.y if g.txt[-1] in "+-"]


def _ydist(tbl, full) -> float:
  "How far a full-width ROW's goals are from the best, over goals only."
  gs = _goals(tbl)
  return ezr.dist((abs(ezr.norm(g, full[g.at]) - g.goal) for g in gs),
                  len(gs))


def _cutoff(train_csv: str) -> float:
  "ydist of the worst row in the best 5% of TRAIN_CSV (step1.best's cut)."
  if train_csv not in _cut:
    tbl = step2._load(train_csv)
    idx = step1.best(train_csv)
    _cut[train_csv] = max(_ydist(tbl, tbl.rows[i]) for i in idx)
  return _cut[train_csv]


def _nearness(tbl, q) -> float:
  "Distance from full-width row Q to its nearest training row, over x."
  return min(ezr.xdist(tbl, r, q) for r in tbl.rows)


def _reference(train_csv: str) -> list:
  "Sorted leave-one-out nearness of a sample of training rows."
  if train_csv not in _ref:
    tbl  = step2._load(train_csv)
    step = max(1, len(tbl.rows) // REF)
    seen = tbl.rows[::step]
    ref  = [min(ezr.xdist(tbl, o, r) for o in tbl.rows if o is not r)
            for r in seen]
    _ref[train_csv] = sorted(ref)
  return _ref[train_csv]


def stream(train_csv: str, rows: list) -> list[bool]:
  "Keep (True) each row whose goals are within the training cut."
  tbl = step2._load(train_csv)
  cut = _cutoff(train_csv)
  out = []
  for row in rows:
    full = step2._full(tbl, row)
    if any(not step2._known(full[g.at]) for g in _goals(tbl)):
      out.append(False)               # a missing goal cannot be judged
    else:
      out.append(_ydist(tbl, full) <= cut)
  return out


def drift(train_csv: str, rows: list) -> int | None:
  "Index of the last row of the first block that no longer looks like TRAIN_CSV, or None."
  tbl  = step2._load(train_csv)
  ref  = _reference(train_csv)
  near = []
  for row in rows:
    near.append(_nearness(tbl, step2._full(tbl, row)))
    if len(near) % WINDOW == 0:
      block = sorted(near[-WINDOW:])
      if not ezr.ks(ref, block, a=ALPHA):
        return len(near) - 1
  return None

