# step8.py
#!/usr/bin/env python3 -B
"Fewest input changes that make a bad build good, leaving frozen columns alone, via ezr."
import itertools, math
import ezr
import step2

MAX_EVALS = 800   # guesses spent on the exhaustive search, all sizes together
CAP       = 8     # numeric options per column: quantiles of the observed values

_cache: dict[str, tuple] = {}   # train_csv -> (tbl, goals, cut, options)

def _goals(tbl) -> list:
  "The goal columns: names ending in + or -."
  return [c for c in tbl.y if c.txt[-1] in "+-"]

def _align(tbl, row: dict) -> list:
  "Turn a {name: value} dict into a list aligned with tbl.names; missing is '?'."
  out = []
  for name in tbl.names:
    v = row.get(name, "?")
    if isinstance(v, str): v = ezr.atom(v)
    out.append(v)
  return out

def _gydist(goals, vals) -> float | None:
  "Distance of VALS' goals from the ideal, or None if a goal cell is unknown."
  if any(vals[c.at] == "?" for c in goals): return None
  return ezr.dist((abs(ezr.norm(c, vals[c.at]) - c.goal) for c in goals),
                  len(goals))

def _options(tbl, col) -> list:
  "Values a column takes in training: all of them, or CAP quantiles if numeric and many."
  vs   = {r[col.at] for r in tbl.rows if r[col.at] != "?"}
  nums = sorted(v for v in vs if isinstance(v, (int, float)))
  syms = sorted((v for v in vs if not isinstance(v, (int, float))), key=str)
  if len(nums) > CAP:
    nums = sorted({nums[round(i * (len(nums) - 1) / (CAP - 1))]
                   for i in range(CAP)})
  return nums + syms

def _load(train_csv: str) -> tuple:
  "Read TRAIN_CSV once; return its table, goals, the good cutoff, and input options."
  if train_csv not in _cache:
    tbl   = ezr.Tbl(ezr.csv(train_csv))
    goals = _goals(tbl)
    ys    = sorted(y for y in (_gydist(goals, r) for r in tbl.rows)
                   if y is not None)
    cut   = ys[max(1, len(ys) // 4) - 1] if ys else None   # best quarter
    options = {c.txt: _options(tbl, c) for c in tbl.x}
    _cache[train_csv] = (tbl, goals, cut, options)
  return _cache[train_csv]

def _dist(train_csv: str, tbl, goals, inputs: dict) -> float:
  "Estimated goal distance of a build with these INPUTS; goals are guessed."
  est  = step2.guess(train_csv, inputs)
  vals = _align(tbl, inputs)
  for g in goals:
    if g.txt not in est: return float("inf")
    vals[g.at] = est[g.txt][0]          # the median guess, as in step6
  d = _gydist(goals, vals)
  return float("inf") if d is None else d

def fix(train_csv, row, frozen=()) -> dict:
  """Inputs of ROW (goals removed) with as few values changed as possible
  so that it becomes good. Columns named in FROZEN never change. Search is
  exhaustive over 1, 2, ... changed columns while the budget allows, then
  greedy. Raises ValueError if no fix is found, or if FROZEN names a column
  the table does not have (so a typo cannot silently unfreeze a column)."""
  tbl, goals, cut, options = _load(train_csv)
  if cut is None:
    raise ValueError("cannot judge: no training row has every goal known")
  if isinstance(frozen, str): frozen = (frozen,)
  frozen = set(frozen)
  unknown = frozen - set(tbl.names)
  if unknown:
    raise ValueError(f"frozen names not in the table: {sorted(unknown)}")
  goal_names = {g.txt for g in goals}
  clean = {k: (ezr.atom(v) if isinstance(v, str) else v)
           for k, v in row.items() if k not in goal_names}
  cur  = _align(tbl, clean)
  cols = [c for c in tbl.x if c.txt not in frozen]
  opts = {c.txt: [v for v in options[c.txt] if v != cur[c.at]] for c in cols}

  spent = [0]
  def dist(changes: dict) -> float:
    "Estimated goal distance after applying CHANGES to the clean inputs."
    spent[0] += 1
    return _dist(train_csv, tbl, goals, {**clean, **changes})

  base = dist({})
  if base <= cut: return dict(clean)   # already good: nothing to flip
  best = (base, {})

  # exhaustive: fewest changed columns first; the best good one at a size wins
  for k in range(1, len(cols) + 1):
    size = sum(math.prod(len(opts[c.txt]) for c in combo)
               for combo in itertools.combinations(cols, k))
    if spent[0] + size > MAX_EVALS: break
    found = None
    for combo in itertools.combinations(cols, k):
      names = [c.txt for c in combo]
      for vals in itertools.product(*(opts[m] for m in names)):
        changes = dict(zip(names, vals))
        d = dist(changes)
        if d < best[0]: best = (d, changes)
        if d <= cut and (found is None or d < found[0]):
          found = (d, changes)
    if found: return {**clean, **found[1]}

  # greedy fallback: keep taking the single change that helps most
  changes, d = {}, base
  while len(changes) < len(cols):
    step = None
    for c in cols:
      if c.txt in changes: continue
      for v in opts[c.txt]:
        td = dist({**changes, c.txt: v})
        if step is None or td < step[0]: step = (td, c.txt, v)
    if step is None or step[0] >= d: break
    d = step[0]
    changes[step[1]] = step[2]
    if d <= cut: return {**clean, **changes}
  if d < best[0]: best = (d, changes)

  raise ValueError(
    "no fix found by the search: no change to the unfrozen inputs makes "
    f"this build good (closest goal distance {best[0]:.2f}, cutoff {cut:.2f})")

