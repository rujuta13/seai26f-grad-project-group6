# step26.py
import ezr

GOOD   = 0.25   # a row is good if its goal score is in the best quarter of training rows
MARGIN = 0.10   # a setting must beat the overall share of good rows by this much
MIN_N  = 10     # a setting needs at least this many rows before we claim anything

def _is_num(v) -> bool:
  "True if V is a real number (a bool is not)."
  return isinstance(v, (int, float)) and not isinstance(v, bool)

def _cell(v):
  "A raw cell as an ezr atom; None, blank and '?' become '?'."
  if v is None: return "?"
  if isinstance(v, str):
    s = v.strip()
    return "?" if s in ("", "?") else ezr.atom(s)
  return v

def _align(tbl: ezr.TBL, row) -> ezr.ROW:
  """Align ROW to TBL's columns. ROW is a dict by column name (with or
  without its + or - sign), or a list over every column or over the
  non-goal columns. Missing is '?'; a number column holding text is '?'."""
  q = ["?"] * len(tbl.names)
  if isinstance(row, dict):
    pairs = [(at, row.get(name, row.get(name.rstrip("+-"))))
             for at, name in enumerate(tbl.names)]
  elif len(row) == len(tbl.names):
    pairs = list(enumerate(row))
  else:
    keep  = [at for at, n in enumerate(tbl.names) if n[-1] not in "+-"]
    pairs = list(zip(keep, row))
  for at, v in pairs: q[at] = _cell(v)
  for c in tbl.cols.values():
    if type(c) is ezr.Num and not _is_num(q[c.at]): q[c.at] = "?"
  return q

def _goals(tbl: ezr.TBL) -> list:
  "The numeric goal columns of TBL: names ending in + or -."
  return [c for c in tbl.y
          if type(c) is ezr.Num and tbl.names[c.at][-1] in "+-"]

def _complete(goals: list, q: ezr.ROW) -> bool:
  "True if every goal cell of Q is a number."
  return all(_is_num(q[c.at]) for c in goals)

def _ydist(goals: list, q: ezr.ROW) -> float:
  "How far Q's goals are from the best they could be: 0 is best, 1 is worst."
  return ezr.dist((abs(ezr.norm(c, q[c.at]) - c.goal) for c in goals),
                  len(goals))

def _usable(tbl: ezr.TBL) -> list:
  """Input columns that can support a checkable reason. Near-unique symbols
  (ids, names, urls) match almost no other row, so they are left out."""
  n = len(tbl.rows)
  return [c for c in tbl.x
          if type(c) is ezr.Num or len(c.has) <= max(1, n // 2)]

def _rate(tbl: ezr.TBL, goals: list):
  """Rows with complete goals, whether each is good, the goal-score cutoff
  for good, and the share of good rows."""
  rows = [r for r in tbl.rows if _complete(goals, r)]
  if not rows: return [], [], 0.0, 0.0
  ys   = [_ydist(goals, r) for r in rows]
  cut  = sorted(ys)[min(len(ys) - 1, int(GOOD * (len(ys) - 1)))]
  good = [y <= cut for y in ys]
  return rows, good, cut, sum(good) / len(good)

def _label(col: ezr.COL, name: str, v) -> str:
  "A readable name for one split of COL."
  if type(col) is ezr.Sym: return f"{name} = {ezr.say(v)}"
  return f"{name} <= {ezr.say(v)}"

def _settings(tbl: ezr.TBL, rows: ezr.ROWS, good: list) -> list:
  """Every usable input split with at least MIN_N rows, as
  (name, label, go, n, share_good)."""
  out = []
  for col in _usable(tbl):
    name = tbl.names[col.at]
    for v, go in ezr.candidates(col, rows):
      hits  = [i for i, r in enumerate(rows) if go(r)]
      n     = len(hits)
      if n < MIN_N: continue
      share = sum(good[i] for i in hits) / n
      out.append((name, _label(col, name, v), go, n, share))
  return out

def _recommend(settings: list, base: float) -> list:
  "For each input, its best setting, if it clearly beats the overall share of good."
  best: dict[str, tuple] = {}
  for s in settings:
    if s[4] > base + MARGIN and (s[0] not in best or s[4] > best[s[0]][4]):
      best[s[0]] = s
  return sorted(best.values(), key=lambda s: -s[4])

def _share(go, rows: ezr.ROWS, good: list):
  "How many rows of ROWS satisfy GO, and what share of those are good (None if none)."
  hits = [i for i, r in enumerate(rows) if go(r)]
  n    = len(hits)
  return n, (sum(good[i] for i in hits) / n if n else None)

def _group(rows: ezr.ROWS, good: list, rec: list):
  "How many rows of ROWS match every recommended setting, and their share good."
  hits = [i for i, r in enumerate(rows) if all(s[2](r) for s in rec)]
  n    = len(hits)
  return n, (sum(good[i] for i in hits) / n if n else None)

def _held(n: int, s) -> str:
  "Text for a held-out count and share."
  return f"{n} rows, {s:.0%} good" if n else "0 rows"

def explain(train_csv) -> str:
  "Plain-words reasons, checkable by hand, for what makes rows good."
  tbl   = ezr.Tbl(ezr.csv(train_csv))
  goals = _goals(tbl)
  if not goals or not tbl.x:
    return ("Cannot explain: the file needs goal columns (names ending + or -) "
            "and input columns.")
  rows, good, cut, base = _rate(tbl, goals)
  if len(rows) < 2 * MIN_N:
    return (f"Cannot explain: only {len(rows)} training rows have every goal "
            f"value; at least {2 * MIN_N} are needed to test reasons on held-out rows.")
  # discovery on every other row; the reasons are then checked on the rest
  fit_rows = rows[0::2]
  fit_good = good[0::2]
  chk_rows = rows[1::2]
  chk_good = good[1::2]
  fit_base = sum(fit_good) / len(fit_good)
  gnames   = ", ".join(tbl.names[c.at] for c in goals)
  settings = _settings(tbl, fit_rows, fit_good)
  rec      = _recommend(settings, fit_base)
  lines = [
    f"1. What 'good' means: a row is good when its goals ({gnames}) are in the "
    f"best {GOOD:.0%} of the {len(rows)} training rows that have every goal value. "
    f"Each row gets a goal score from 0 (best possible on every goal) to 1 (worst). "
    f"Good means a score of {cut:.2f} or less. In this file {base:.0%} of rows are good.",
    f"2. Recommended settings. Found on half the rows (where {fit_base:.0%} are good "
    f"overall), each input's setting whose rows are good most often, if it beats "
    f"that share by at least {MARGIN * 100:.0f} points and has at least {MIN_N} rows. "
    f"The other half checks them:",
  ]
  if rec:
    for _, label, go, n, share in rec:
      cn, cs = _share(go, chk_rows, chk_good)
      lines.append(f"   - {label}: {n} rows, {share:.0%} good. "
                   f"Held-out: {_held(cn, cs)}. "
                   f"To check: filter the file to these rows and count how many "
                   f"are in the best {GOOD:.0%} on the goals.")
    dn, ds = _group(fit_rows, fit_good, rec)
    cn, cs = _group(chk_rows, chk_good, rec)
    if dn >= MIN_N and cn >= MIN_N:
      lines.append(f"   Rows matching all of these together: {dn} found, {ds:.0%} good; "
                   f"{cn} held-out, {cs:.0%} good.")
    else:
      lines.append(f"   Rows matching all of these together: {dn} found, {cn} held-out; "
                   f"too few to trust, so the combined effect is untested.")
  else:
    lines.append("   No single setting beats the overall share by enough, "
                 "so no setting is recommended from this data.")
  weak = sorted((s for s in settings if s[4] < fit_base - MARGIN),
                key=lambda s: s[4])[:3]
  lines.append("3. Settings linked to weaker rows (lowest share good):")
  for _, label, go, n, share in weak:
    cn, cs = _share(go, chk_rows, chk_good)
    lines.append(f"   - {label}: {n} rows, {share:.0%} good. Held-out: {_held(cn, cs)}.")
  if not weak: lines.append("   None clear enough to report.")
  lines.append("4. Limits: these are patterns in one file of past rows, not proof "
               "that a setting causes better goals. Rows with a setting may differ "
               f"in other ways. Settings with fewer than {MIN_N} rows were left out. "
               "The held-out half comes from the same file, so a fresh sample of "
               "rows is the real test. Before acting, check a sample of rows by hand "
               "against the goal columns.")
  return "\n".join(lines)

def why(train_csv, row) -> str:
  "One or two sentences on why this row (goals included) is good or bad."
  tbl   = ezr.Tbl(ezr.csv(train_csv))
  goals = _goals(tbl)
  if not goals or not tbl.x:
    return "Cannot rate this row: the file needs goal columns and input columns."
  rows, good, cut, base = _rate(tbl, goals)
  if not rows:
    return "Cannot rate this row: no training row has numeric values for every goal."
  q        = _align(tbl, row)
  settings = _settings(tbl, rows, good)
  rec      = _recommend(settings, base)
  is_good  = None
  if not _complete(goals, q):
    missing = [tbl.names[c.at] for c in goals if not _is_num(q[c.at])]
    head = (f"Cannot rate it as good or bad: its goal values are missing "
            f"({', '.join(missing)}).")
  else:
    y       = _ydist(goals, q)
    is_good = y <= cut
    if is_good:
      head = (f"Good: its goals are in the best {GOOD:.0%} of training rows "
              f"(goal score {y:.2f} <= {cut:.2f}, where 0 is best).")
    else:
      head = (f"Bad: its goals fall outside the best {GOOD:.0%} of training rows "
              f"(goal score {y:.2f} > {cut:.2f}, where 0 is best).")
  if not rec:
    tail = ("No input setting clearly separates good rows here, "
            "so its inputs give no reason either way.")
  else:
    met   = [s[1] for s in rec if s[2](q)]
    lacks = [s[1] for s in rec if not s[2](q)]
    tail  = (f"Its inputs match {len(met)} of the {len(rec)} recommended settings")
    if lacks: tail += f"; it lacks {', '.join(lacks[:2])}"
    if is_good is False:
      weak_hit = [s[1] for s in settings if s[4] < base - MARGIN and s[2](q)]
      if weak_hit: tail += f"; it has {weak_hit[0]}, which is linked to weaker rows"
    tail += "."
  return f"{head} {tail}"

