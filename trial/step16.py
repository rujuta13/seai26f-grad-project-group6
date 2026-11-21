# step16.py
"Step 16: justify, for the client. Plain reasons a manager can check, and a short verdict on one player."
import math
import os
import ezr

QUANTILE = 0.25   # a player is "good" if its goals are in the best 25% of train
TOP      = 5      # settings to report in explain, per direction
MINSUP   = 5      # a setting must occur in at least this many players
LIFT     = 1.2    # ...and be this much more (or less) common among good players

_CACHE: dict = {}  # (path, mtime) -> (table, pairs, distances, cutoff, good rate, settings)

def _key(train_csv: str) -> tuple:
  "Identity of TRAIN_CSV: its absolute path and when it last changed."
  path = os.path.abspath(os.path.expanduser(train_csv))
  return (path, os.path.getmtime(path))

def _isnum(v) -> bool:
  "True for a usable finite number."
  return (isinstance(v, (int, float)) and not isinstance(v, bool)
          and math.isfinite(v))

def _name(s: str) -> str:
  "A column name without its +, - or ! suffix."
  return s[:-1] if s[-1:] in ("+", "-", "!") else s

def _coerce(v):
  "Strings become atoms; other values pass through."
  return ezr.atom(v) if isinstance(v, str) else v

def _full(tbl, row) -> list:
  """Lift ROW into a full-width row. Accepts a {column name: value} dict,
  a full-width list, or a list with goal columns removed (header order).
  Any cell not given is '?'. Non-numeric cells in numeric columns become '?'."""
  names = tbl.names
  out   = ["?"] * len(names)
  if isinstance(row, dict):
    for at, n in enumerate(names):
      if n in row:
        out[at] = _coerce(row[n])
      elif n[-1:] in ("+", "-", "!") and _name(n) in row:
        out[at] = _coerce(row[_name(n)])
  else:
    row = list(row)
    if len(row) == len(names):
      out = [_coerce(v) for v in row]
    else:
      keep = [i for i, n in enumerate(names) if n[-1:] not in ("+", "-", "!")]
      for i, v in zip(keep, row): out[i] = _coerce(v)
  for c in tbl.cols.values():
    if type(c) is ezr.Num and not _isnum(out[c.at]):
      out[c.at] = "?"
  return out

def _ydist(tbl, row) -> float | None:
  "Distance of ROW's numeric goals from their best; None if no goal is known."
  sq, n = 0.0, 0
  for c in tbl.y:
    if type(c) is not ezr.Num: continue
    v = row[c.at]
    if not _isnum(v): continue
    sq += (ezr.norm(c, v) - c.goal) ** 2
    n  += 1
  return (sq / n) ** .5 if n else None

def _pct(xs: list, p: float) -> float:
  "Percentile P (0..1) of sorted XS, by linear interpolation."
  i  = p * (len(xs) - 1)
  lo = int(i)
  hi = min(lo + 1, len(xs) - 1)
  return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)

def _settings(tbl, pairs: list, base: float) -> list:
  """Every checkable input setting, with how many players have it and how
  many of those are good. Each is a dict with at, text, test, n, rate, lift."""
  out = []

  def add(at, text, test):
    flags = [g for r, g in pairs if r[at] != "?" and test(r[at])]
    if len(flags) < MINSUP or base == 0: return
    rate = sum(flags) / len(flags)
    out.append({"at": at, "text": text, "test": test, "n": len(flags),
                "rate": rate, "lift": rate / base})

  for col in tbl.x:
    if type(col) is ezr.Sym:
      if len(col.has) >= col.n: continue     # every value distinct: an identifier
      for v in sorted({r[col.at] for r, _ in pairs if r[col.at] != "?"}, key=str):
        add(col.at, f"{col.txt} = {v}", lambda x, v=v: x == v)
    elif type(col) is ezr.Num:
      vals = sorted(r[col.at] for r, _ in pairs if _isnum(r[col.at]))
      if len(vals) < 2 * MINSUP: continue
      lo, hi = _pct(vals, 0.25), _pct(vals, 0.75)
      add(col.at, f"{col.txt} at or above {ezr.say(hi)}",
          lambda x, t=hi: _isnum(x) and x >= t)
      add(col.at, f"{col.txt} at or below {ezr.say(lo)}",
          lambda x, t=lo: _isnum(x) and x <= t)
  return out

def _model(train_csv: str):
  """Train table; (row, is_good) for every row with known goals; the sorted
  goal distances; the good cutoff; the good rate; and the input settings.
  Cached per file version."""
  key = _key(train_csv)
  if key not in _CACHE:
    tbl    = ezr.Tbl(ezr.csv(train_csv))
    scored = []
    for r in tbl.rows:
      d = _ydist(tbl, r)
      if d is not None: scored.append((r, d))
    ys    = sorted(d for _, d in scored)
    thr   = ys[int(QUANTILE * (len(ys) - 1))] if ys else None
    pairs = [(r, d <= thr) for r, d in scored]
    base  = sum(g for _, g in pairs) / len(pairs) if pairs else 0.0
    sets  = _settings(tbl, pairs, base)
    _CACHE[key] = (tbl, pairs, ys, thr, base, sets)
  return _CACHE[key]

def _goal_text(c) -> str:
  "Goal name without its suffix, and its direction."
  return f"{_name(c.txt)} ({'maximise' if c.goal else 'minimise'})"

def _line(s: dict, base: float) -> str:
  "One checkable line: the setting, its good rate, and the overall good rate."
  return (f"{s['text']}: {s['rate']:.0%} of the {s['n']} players with it are "
          f"good, versus {base:.0%} overall.")

def explain(train_csv) -> str:
  "Plain reasons for what makes players good; each one is a count a skeptical expert can check."
  tbl, pairs, ys, thr, base, sets = _model(train_csv)
  if not ys:
    return "No training player has known goal values, so nothing can be explained."
  goals = ", ".join(_goal_text(c) for c in tbl.y if type(c) is ezr.Num)
  out   = [f"A player is good when their goals ({goals}) are in the best "
           f"{round(100 * QUANTILE)}% of the {len(ys)} training players with "
           f"known goals. Distance 0 is ideal; good means distance at or below "
           f"{ezr.say(thr)}, and the average distance over all of them is "
           f"{ezr.say(sum(ys) / len(ys))}."]

  up   = sorted((s for s in sets if s["lift"] >= LIFT),
                key=lambda s: (-s["lift"], -s["n"], s["text"]))
  down = sorted((s for s in sets if s["lift"] <= 1 / LIFT),
                key=lambda s: (s["lift"], -s["n"], s["text"]))
  if up:
    out.append("Settings more common in good players (checkable by counting):")
    out += [f"  {_line(s, base)}" for s in up[:TOP]]
  else:
    out.append("No setting is clearly more common in good players, so no "
               "positive rule is claimed.")
  if down:
    out.append("Settings less common in good players (checkable by counting):")
    out += [f"  {_line(s, base)}" for s in down[:TOP]]
  out.append("These are associations in the training data, not proof that a "
             "setting causes the goals; check each on video before relying on it.")
  return "\n".join(out)

def why(train_csv, row) -> str:
  "One or two sentences: is this player good or bad, judged by its goals, and one checkable reason from its settings."
  tbl, pairs, ys, thr, base, sets = _model(train_csv)
  if not ys:
    return ("This player cannot be rated: the training file has no known "
            "goal values.")
  full  = _full(tbl, row)
  known = [(c, full[c.at]) for c in tbl.y
           if type(c) is ezr.Num and _isnum(full[c.at])]
  if not known:
    return "This player cannot be rated: it gives no numeric goal values."
  d       = _ydist(tbl, full)
  verdict = "good" if d <= thr else "bad"
  beat    = sum(1 for y in ys if y > d) / len(ys)
  vals    = ", ".join(f"{_name(c.txt)}={ezr.say(v)}" for c, v in known)
  first   = (f"This player is {verdict}: their goals ({vals}) sit closer to the "
             f"ideal than {beat:.0%} of the {len(ys)} training players "
             f"(distance {ezr.say(d)}, good means at or below {ezr.say(thr)}).")

  def has(s) -> bool:
    cell = full[s["at"]]
    return cell != "?" and s["test"](cell)

  def lacks(s) -> bool:
    cell = full[s["at"]]
    return cell != "?" and not s["test"](cell)

  def phrase(s) -> str:
    return (f"{s['text']} ({s['rate']:.0%} of the {s['n']} players with that "
            f"setting are good, versus {base:.0%} overall)")

  if verdict == "good":
    verb  = "has"
    picks = [s for s in sets if s["lift"] >= LIFT and has(s)]
  else:
    verb  = "has"
    picks = [s for s in sets if s["lift"] <= 1 / LIFT and has(s)]
    if not picks:
      verb  = "lacks"
      picks = [s for s in sets if s["lift"] >= LIFT and lacks(s)]
  picks.sort(key=lambda s: (-abs(s["lift"] - 1), -s["n"], s["text"]))

  if not picks:
    return (f"{first} No single setting in its inputs clearly separates good "
            f"players from bad ones, so the verdict rests on its goal numbers alone.")
  reasons = " and ".join(phrase(s) for s in picks[:2])
  return f"{first} Checkable reasons: it {verb} {reasons}."

