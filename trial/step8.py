# step8.py
import step2
import step4

MAX_VALS = 5   # candidate values tried per input column

def _values(tbl, c) -> list:
  "Candidate values for input column C: its distinct train values, thinned to MAX_VALS."
  vs = sorted({r[c.at] for r in tbl.rows if r[c.at] != "?"},
              key=lambda v: (isinstance(v, str), v))
  if len(vs) > MAX_VALS:                       # spread the picks across the range
    step = len(vs) / MAX_VALS
    vs   = [vs[int(i * step)] for i in range(MAX_VALS)]
  return vs

def _est_score(train_csv, tbl, score, full) -> float:
  "Estimate how good FULL (goals missing) is: fill goals with step2.guess, then score."
  est = step2.guess(train_csv, full)
  g   = list(full)
  for c in tbl.y: g[c.at] = est[c.txt][0]   # the estimated goal values
  return score(g)                           # >0 means better than the train average

def fix(train_csv, row, frozen=()) -> dict:
  """Inputs of ROW, changed as few as possible (greedily, one flag at a time)
  so that it becomes good. Columns named in FROZEN never change."""
  tbl, score = step4._scorer(train_csv)
  frozen = set(frozen)
  unknown = frozen - set(tbl.names)
  if unknown: raise ValueError(f"fix: unknown column(s) {sorted(unknown)}")

  full = step2._row(tbl, row)               # ROW in full layout, goals "?"
  free = [c for c in tbl.x if tbl.names[c.at] not in frozen]
  cands = {c.at: _values(tbl, c) for c in free}

  cur = list(full)
  now = _est_score(train_csv, tbl, score, cur)
  changed = set()
  while now <= 0:                           # still not good: flip one more flag
    best = None                             # (score, at, value)
    for c in free:
      if c.at in changed: continue
      for v in cands[c.at]:
        if v == cur[c.at]: continue
        trial = list(cur); trial[c.at] = v
        s = _est_score(train_csv, tbl, score, trial)
        if best is None or s > best[0]: best = (s, c.at, v)
    if best is None or best[0] <= now: break   # no single flip helps: stop
    now, at, v = best
    cur[at] = v
    changed.add(at)

  return {tbl.names[c.at]: cur[c.at] for c in tbl.x}

