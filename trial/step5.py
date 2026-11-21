# step5.py
import ezr
import step3
import step4

def blame(train_csv: str) -> list[str]:
  "Input column names, most to blame for bad rows (bad on the goals) first."
  tbl, score = step4._scorer(train_csv)
  ws    = [score(r) for r in tbl.rows]
  bad   = [r for r, w in zip(tbl.rows, ws) if w < 0]   # worse than average
  good  = [r for r, w in zip(tbl.rows, ws) if w > 0]   # better than average
  names = [tbl.names[c.at] for c in tbl.x]
  if not bad or not good: return names                 # nothing to contrast

  blamed = []
  for i, c in enumerate(tbl.x):
    at = c.at
    b  = [r[at] for r in bad  if r[at] != "?"]
    g  = [r[at] for r in good if r[at] != "?"]
    if not b or not g:
      s = 0.0
    elif type(c) is ezr.Sym:
      # how differently the symbols are spread over bad vs good rows (0..1)
      cb, cg = {}, {}
      for v in b: cb[v] = cb.get(v, 0) + 1
      for v in g: cg[v] = cg.get(v, 0) + 1
      s = 0.5 * sum(abs(cb.get(v, 0)/len(b) - cg.get(v, 0)/len(g))
                    for v in set(cb) | set(cg))
    else:
      # squashed numbers (0..1); a gap that step3 calls noise counts as zero
      nb = [ezr.norm(c, v) for v in b]
      ng = [ezr.norm(c, v) for v in g]
      s  = 0.0 if step3.same(nb, ng) else \
           abs(sum(nb)/len(nb) - sum(ng)/len(ng))
    blamed.append((s, i))

  order = sorted(blamed, key=lambda t: (-t[0], t[1]))
  return [names[i] for _, i in order]

