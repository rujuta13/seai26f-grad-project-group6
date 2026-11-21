# step8.py
"""
step8: fix.  Given a bad row (goals removed), change as few input values as
possible so that its predicted goals become good.  Columns named in FROZEN
are never changed.  Reuses step2 (model, _full), and step4 (_train_ys,
_ydist, KEEP_FRAC) so that "good" means what it means everywhere else.
"""
from itertools import combinations, product
import step2
import step4


def fix(train_csv, row, frozen=()) -> dict:
  """The same row's inputs, with the fewest values changed so that it is
  predicted good. Columns in FROZEN never change. If no change makes the
  row good, return the lowest-ydist version found by the search."""
  tbl, _ = step2._model(train_csv)
  frozen = set(frozen)
  unknown = frozen - set(tbl.names)
  if unknown:
    raise ValueError(f"fix: frozen names not in the table: {sorted(unknown)}")

  base = step2._full(tbl, row)
  for c in tbl.y:                     # goals are unknown: predict them
    base[c.at] = "?"

  ys  = step4._train_ys(train_csv)
  k   = max(1, int(step4.KEEP_FRAC * len(ys)))
  bar = ys[k - 1]                     # good means ydist <= bar

  mut = [c for c in tbl.x if c.txt not in frozen]
  options = []                        # (column position, candidate values)
  for c in mut:
    cur  = base[c.at]
    vals = [v for v in dict.fromkeys(r[c.at] for r in tbl.rows
                                     if r[c.at] != "?")
            if v != cur]
    options.append((c.at, vals))

  best_y, best_full = None, None      # lowest ydist seen, over all tries
  found = None                        # lowest ydist among good tries
  for depth in range(len(mut) + 1):   # fewest changes first
    for cols in combinations(options, depth):
      for vals in product(*(vs for _, vs in cols)):
        cand = list(base)
        for (at, _), v in zip(cols, vals):
          cand[at] = v
        y = step4._ydist(train_csv, cand)
        if best_y is None or y < best_y:
          best_y, best_full = y, cand
        if y <= bar and (found is None or y < found[0]):
          found = (y, cand)
    if found is not None:             # no more changes needed
      best_full = found[1]
      break

  return {c.txt: best_full[c.at] for c in tbl.x}

