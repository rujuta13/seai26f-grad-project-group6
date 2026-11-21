# step28.py
import ezr

GOOD = 0.25   # a row is good if its ydist is in the best quarter of training rows

def fix(train_csv, row, frozen=()) -> dict:
  """The inputs of ROW (a bad row, goals removed), with as few values changed
  as possible so it becomes good. Each candidate is a training row whose goals
  are known, so every fix is a configuration observed to be good. Columns
  named in FROZEN keep their values. Returns {input name: value} over the
  table's input columns."""

  def is_num(v) -> bool:
    "True if V is a real number (a bool is not)."
    return isinstance(v, (int, float)) and not isinstance(v, bool)

  def cell(v):
    "A raw cell as an ezr atom; None, blank and '?' become '?'."
    if v is None: return "?"
    if isinstance(v, str):
      s = v.strip()
      return "?" if s in ("", "?") else ezr.atom(s)
    return v

  def out(v):
    "Numbers as floats; everything else unchanged."
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return float(v)
    return v

  def ydist(r) -> float:
    "How far row R's numeric goals are from the best they could be (0 is best)."
    return ezr.dist((abs(ezr.norm(c, r[c.at]) - c.goal) for c in goals),
                    len(goals))

  tbl    = ezr.Tbl(ezr.csv(train_csv))
  names  = tbl.names
  goals  = [c for c in tbl.y if type(c) is ezr.Num]
  ins    = [c.at for c in tbl.x]
  if frozen is None:
    frozen = set()
  elif isinstance(frozen, str):
    frozen = {frozen}
  else:
    frozen = set(frozen)
  unknown = frozen - set(names)
  if unknown: raise ValueError(f"fix: no column named {sorted(unknown)}")
  fixed  = {at for at in ins if names[at] in frozen}

  # the query, aligned to every column; goals (if given) are ignored
  q = ["?"] * len(names)
  if isinstance(row, dict):
    pairs = [(at, row.get(name, row.get(name.rstrip("+-"))))
             for at, name in enumerate(names)]
  elif len(row) == len(names):
    pairs = list(enumerate(row))
  else:
    keep  = [at for at, n in enumerate(names) if n[-1] not in "+-"]
    pairs = list(zip(keep, row))
  for at, v in pairs: q[at] = cell(v)

  # training rows with every goal known, and the ydist cutoff for good
  def done(r) -> bool:
    "True if every goal cell of R is a number."
    return all(is_num(r[c.at]) for c in goals)

  full = [r for r in tbl.rows if goals and done(r)]
  if not full:
    return {names[at]: out(q[at]) for at in ins}
  ys  = [ydist(r) for r in full]
  cut = sorted(ys)[min(len(ys) - 1, int(GOOD * (len(ys) - 1)))]

  # candidates: known rows that agree with every frozen input, and whose
  # changed values are real (never a blank "?")
  cands = []
  for y, g in zip(ys, full):
    if any(g[at] != q[at] for at in fixed): continue
    diff = [at for at in ins if at not in fixed and g[at] != q[at]]
    if any(g[at] == "?" for at in diff): continue
    cands.append((y, len(diff), g))

  good = [c for c in cands if c[0] <= cut]
  if good:                  # fewest changed values first, then lowest ydist
    _, _, g = min(good, key=lambda c: (c[1], c[0]))
  elif cands:               # no fix reaches good: the closest one we can get
    _, _, g = min(cands, key=lambda c: (c[0], c[1]))
  else:                     # no known row fits the frozen columns: no change
    g = q

  return {names[at]: out(g[at]) for at in ins}

