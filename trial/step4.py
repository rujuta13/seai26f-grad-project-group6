# step4.py
"""
step4: flow.  Rows arrive one at a time.  stream() keeps the ones worth
keeping (good on the goals).  drift() warns when the new rows stop looking
like the training rows.  Reuses step2 (table, tree, _full, guess), step3
(same), and ezr (ydist, same).  Each decision uses only train_csv and the
rows seen so far.
"""
import ezr
import step2
import step3

KEEP_FRAC = 0.05   # keep rows at least as good as the best 5% of training
WINDOW    = 20     # drift: the last WINDOW rows are tested against training

_ys_cache: dict = {}   # train_csv -> sorted training ydists


def _train_ys(train_csv: str) -> list[float]:
  "Sorted ydist of every training row, measured with training's stats."
  if train_csv not in _ys_cache:
    tbl, _ = step2._model(train_csv)
    _ys_cache[train_csv] = sorted(ezr.ydist(tbl, r) for r in tbl.rows)
  return _ys_cache[train_csv]


def _ydist(train_csv: str, row) -> float:
  "How far ROW's goals are from ideal, using training's stats."
  tbl, _ = step2._model(train_csv)
  full = step2._full(tbl, row)
  miss = [c for c in tbl.y if full[c.at] == "?"]
  if miss:                                  # unknown goals: fill by guess
    g = step2.guess(train_csv, row)
    for c in miss:
      v = g[c.txt][0]
      full[c.at] = c.mu if v != v else v    # nan -> training mean
  return ezr.ydist(tbl, full)


def stream(train_csv: str, rows) -> list[bool]:
  "Keep (True) each row whose goals are as good as the best training rows."
  ys  = _train_ys(train_csv)
  k   = max(1, int(KEEP_FRAC * len(ys)))
  bar = ys[k - 1]
  return [_ydist(train_csv, row) <= bar for row in rows]


def drift(train_csv: str, rows) -> int | None:
  "Index of the first row where the recent rows stop matching training."
  base = _train_ys(train_csv)
  seen: list[float] = []
  for i, row in enumerate(rows):
    seen.append(_ydist(train_csv, row))
    recent = seen[-WINDOW:]
    if len(recent) == WINDOW and not step3.same(base, recent):
      return i
  return None

