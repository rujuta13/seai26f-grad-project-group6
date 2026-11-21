# step25.py
import ezr
import step2
import step3

# The client's two questions share one split of the customers:
#   bad  = never responds (RESPONSE+ is 0) or complained (COMPLAIN- is 1)
#   good = responded (RESPONSE+ is 1) and did not complain
# Inputs are ranked by how differently they are spread over bad vs good rows.
# RESPONSE is the last campaign only, so "never responds" here means "did not
# respond to the last campaign". Rows that are not clearly bad or good are left out.

def _at(tbl, base: str) -> int:
  "Column index of the goal named BASE (its name without the + or - suffix)."
  for at, s in enumerate(tbl.names):
    if s.rstrip("+-") == base: return at
  raise ValueError(f"blame: no {base} column in the header")

def _groups(tbl, resp_at: int, comp_at: int) -> tuple[list, list]:
  "Rows that are good (responded, no complaint) and bad (no response, or complained)."
  good, bad = [], []
  for r in tbl.rows:
    resp, comp = r[resp_at], r[comp_at]
    if resp == 0 or comp == 1:
      bad.append(r)
    elif resp == 1 and comp == 0:
      good.append(r)
  return good, bad

def _contrast(c, b: list, g: list) -> float:
  "How differently column C is spread over bad values B and good values G (0 = no difference)."
  if not b or not g: return 0.0
  if type(c) is ezr.Sym:
    cb, cg = {}, {}
    for v in b: cb[v] = cb.get(v, 0) + 1
    for v in g: cg[v] = cg.get(v, 0) + 1
    return 0.5 * sum(abs(cb.get(v, 0)/len(b) - cg.get(v, 0)/len(g))
                     for v in set(cb) | set(cg))
  # squashed numbers (0..1); a gap that step3 calls noise counts as zero
  nb = [ezr.norm(c, v) for v in b]
  ng = [ezr.norm(c, v) for v in g]
  return 0.0 if step3.same(nb, ng) else \
         abs(sum(nb)/len(nb) - sum(ng)/len(ng))

def blame(train_csv: str) -> list[str]:
  "Input column names, most to blame for bad rows (bad on the goals) first."
  tbl = step2._table(train_csv)
  if not tbl.rows: raise ValueError(f"no rows in {train_csv}")
  resp_at = _at(tbl, "RESPONSE")
  comp_at = _at(tbl, "COMPLAIN")
  good, bad = _groups(tbl, resp_at, comp_at)
  names = [tbl.names[c.at] for c in tbl.x]
  if not good or not bad: return names          # nothing to contrast

  scored = []
  for i, c in enumerate(tbl.x):
    b = [r[c.at] for r in bad  if r[c.at] != "?"]
    g = [r[c.at] for r in good if r[c.at] != "?"]
    scored.append((_contrast(c, b, g), i))

  order = sorted(scored, key=lambda t: (-t[0], t[1]))
  return [names[i] for _, i in order]

