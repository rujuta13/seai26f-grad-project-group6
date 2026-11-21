# step6.py
"""
step6: justify.  Two plain answers a skeptic can check.  explain() says
what the good rows share, with the numbers behind it.  why() says why one
row is good or bad.  Reuses step2 (model, _full, _q), step3 (same),
step4 (_train_ys, _ydist, KEEP_FRAC), step5 (blame, BAD_FRAC), and
ezr (tree, leaf, leafs, ydist, Sym, say).
"""
import bisect
import ezr
import step2
import step3
import step4
import step5


def _test(tbl, node, yes: bool) -> str:
  "Plain words for one side of NODE's cut."
  name, v = tbl.names[node.at], ezr.say(node.v)
  if type(tbl.cols[node.at]) is ezr.Sym:
    return f"{name} = {v}" if yes else f"{name} != {v}"
  return f"{name} <= {v}" if yes else f"{name} > {v}"


def _path(tbl, node, target) -> list[str] | None:
  "The cuts from NODE down to TARGET, as plain words; None if absent."
  if node is target: return []
  for yes, kid in zip((True, False), node.kids):
    rest = _path(tbl, kid, target)
    if rest is not None:
      return [_test(tbl, node, yes)] + rest
  return None


def _words(path: list[str]) -> str:
  "A path as English: 'all training rows' if empty."
  return " and ".join(path) if path else "all training rows"


def _family(node):
  "The leaf with the lowest mean ydist, among leaves big enough to trust."
  ls  = ezr.leafs(node)
  big = [z for z in ls if len(z.rows) >= 5] or ls
  return min(big, key=lambda z: z.mu)


def explain(train_csv: str) -> str:
  "What the good rows share, in numbers a skeptic can recheck."
  tbl, node = step2._model(train_csv)
  ys = step4._train_ys(train_csv)               # sorted training ydists
  n  = len(ys)
  if not n: return "No training rows, so nothing to explain."
  k   = max(1, int(step4.KEEP_FRAC * n))
  fam = _family(node)
  inside = {id(r) for r in fam.rows}
  rest   = [r for r in tbl.rows if id(r) not in inside]

  out = [
    "ydist is the distance from ideal on the goals, scaled per goal; 0 is best, lower is better.",
    f"Good rows: the best {k} of {n} training rows, ydist <= {ys[k-1]:.2f}.",
    "Recommended flag set: " + _words(_path(tbl, node, fam) or []) + ".",
  ]

  for c in tbl.y:
    want    = "higher" if c.goal else "lower"
    vs_fam  = sorted(r[c.at] for r in fam.rows if r[c.at] != "?")
    vs_all  = sorted(r[c.at] for r in tbl.rows if r[c.at] != "?")
    if vs_fam and vs_all:
      out.append(f"{c.txt} (want {want}): median "
                 f"{step2._q(vs_fam, 0.5):.2f} in the family, "
                 f"{step2._q(vs_all, 0.5):.2f} overall.")

  fy = [ezr.ydist(tbl, r) for r in fam.rows]
  ry = [ezr.ydist(tbl, r) for r in rest]
  if fy and ry:
    gap = ("is a real gap" if not step3.same(fy, ry)
           else "could be run-to-run noise")
    out.append(f"Family ydist averages {sum(fy)/len(fy):.2f} vs "
               f"{sum(ry)/len(ry):.2f} for the other rows: the gap {gap}.")

  out.append("Inputs most tied to bad rows, most first: "
             + ", ".join(step5.blame(train_csv)[:3]) + ".")
  out.append("Caveat: the family was chosen on training rows; "
             "confirm it on rows the tree never saw.")
  return "\n".join(out)


def why(train_csv: str, row) -> str:
  "One or two sentences: is this row good or bad, and why."
  tbl, node = step2._model(train_csv)
  ys   = step4._train_ys(train_csv)
  n    = len(ys)
  full = step2._full(tbl, row)
  miss = [c.txt for c in tbl.y if full[c.at] == "?"]
  y    = step4._ydist(train_csv, row)
  k    = max(1, int(step4.KEEP_FRAC * n))
  bar  = ys[k-1]
  worst = ys[min(n-1, int((1 - step5.BAD_FRAC) * n))]
  verdict = ("good" if y <= bar else
             "bad" if y >= worst else "middling")
  rank = bisect.bisect_right(ys, y)             # training rows at or below y
  note = (f" (goals {', '.join(miss)} were not given, so they are estimated)"
          if miss else "")
  leaf = ezr.leaf(node, full)
  fam  = _words(_path(tbl, node, leaf) or [])
  first = (f"This row is {verdict}{note}: its ydist is {y:.2f}, at or below "
           f"{rank} of {n} training rows (good: ydist <= {bar:.2f}; "
           f"bad: ydist >= {worst:.2f}).")
  second = (f"It falls in the family where {fam}, which averages "
            f"{leaf.mu:.2f} over {len(leaf.rows)} training rows.")
  return first + " " + second

