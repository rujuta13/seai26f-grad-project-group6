# step10.py
#!/usr/bin/env python3 -B
"""Race our tuner (step7) and guesser (step2) against rivals on test rows, via ezr.

The standard library ships no tuner or regressor, so the rivals are built from
it. 'random_search' labels BUDGET rows at random. 'hillclimb' is a greedy local
search: it labels the unlabelled row nearest the current best. 'guess_linear' is
ordinary least squares on one-hot inputs. Every score is lower-is-better. Tuner
scores are the goal distance of the build picked (0 is ideal). Guesser scores are
the mean absolute error on each goal, in training standard deviations, averaged
over goals and test rows. Labels come only from test_csv."""
import os, random, tempfile
from collections.abc import Callable
import ezr
import step2
import step7

BUDGET = 30     # labels each tuner may take, as in step7
LAMBDA = 1e-6   # ridge term, so the normal equations stay solvable

def _goals(tbl) -> list:
  "The goal columns: names ending in + or -."
  return [c for c in tbl.y if c.txt[-1] in "+-"]

def _gydist(goals, row) -> float:
  "Distance of ROW's goals from the ideal, as ezr measures it."
  return ezr.dist((abs(ezr.norm(c, row[c.at]) - c.goal) for c in goals),
                  len(goals))

def _best_labelled(goals, known: dict) -> int:
  "Index of the row in KNOWN (index -> {goal: value}) nearest the ideal, judged by its labels alone."
  cols = {c.at: ezr.Num(c.txt, c.at) for c in goals}
  for cells in known.values():
    for c in goals: ezr.add(cols[c.at], cells[c.txt])
  def d(i: int) -> float:
    cells = known[i]
    if any(cells[c.txt] == "?" for c in goals): return float("inf")
    return ezr.dist((abs(ezr.norm(cols[c.at], cells[c.txt]) - cols[c.at].goal)
                     for c in goals), len(goals))
  return min(known, key=lambda i: (d(i), i))

def _blind(tbl, goals, path: str) -> None:
  "Write TBL to PATH with every goal cell '?', so the tuner must label to see them."
  hidden = {c.at for c in goals}
  with open(path, "w", encoding="utf-8") as f:
    f.write(",".join(str(n) for n in tbl.names) + "\n")
    for row in tbl.rows:
      f.write(",".join("?" if i in hidden else str(v)
                       for i, v in enumerate(row)) + "\n")

def _hillclimb(full, goals, label, budget: int, rng) -> int:
  "Greedy local search: from a random row, keep labelling the unlabelled row nearest the best so far."
  rows  = full.rows
  first = rng.randrange(len(rows))
  known = {first: label(first)}
  while len(known) < budget:
    best = rows[_best_labelled(goals, known)]
    nxt  = min((i for i in range(len(rows)) if i not in known),
               key=lambda i: ezr.xdist(full, rows[i], best))
    known[nxt] = label(nxt)
  return _best_labelled(goals, known)

def _race_tuners(full, goals, rng) -> dict[str, float]:
  "Goal distance of the build each tuner picks, labelling at most BUDGET rows."
  n      = len(full.rows)
  budget = min(BUDGET, n)
  def label(i: int) -> dict:
    return {c.txt: full.rows[i][c.at] for c in goals}
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "blind.csv")
    _blind(full, goals, path)
    ours = step7.choose(path, label, budget=budget)
  picks = {
    "ours":          ours,
    "random_search": _best_labelled(goals, {i: label(i) for i in
                                            rng.sample(range(n), budget)}),
    "hillclimb":     _hillclimb(full, goals, label, budget, rng),
  }
  return {k: _gydist(goals, full.rows[i]) for k, i in picks.items()}

def _solve(A: list, b: list) -> list[float]:
  "Solve A w = b by Gauss-Jordan elimination with partial pivoting."
  n = len(b)
  M = [row + [b[i]] for i, row in enumerate(A)]
  for k in range(n):
    p = max(range(k, n), key=lambda r: abs(M[r][k]))
    M[k], M[p] = M[p], M[k]
    pv = M[k][k]
    if abs(pv) < 1e-12: continue
    M[k] = [v / pv for v in M[k]]
    for r in range(n):
      if r != k and M[r][k] != 0:
        f = M[r][k]
        M[r] = [a - f * v for a, v in zip(M[r], M[k])]
  return [M[i][n] for i in range(n)]

def _linear(train) -> Callable[[list], dict[str, float]]:
  "Ordinary least squares, one model per goal; symbols one-hot; ridge-stabilised."
  xs     = list(train.x)
  levels = {c.at: sorted(c.has, key=str) for c in xs if type(c) is ezr.Sym}
  def feats(row: list) -> list[float]:
    out = [1.0]                                   # intercept
    for c in xs:
      v = row[c.at]
      if type(c) is ezr.Sym:
        out += [1.0 if v == lv else 0.0 for lv in levels[c.at]]
      else:
        try:    out.append(float(v))
        except (TypeError, ValueError): out.append(float(c.mu))  # unknown: mean
    return out
  models = {}
  for g in _goals(train):
    pairs = [(feats(r), float(r[g.at])) for r in train.rows if r[g.at] != "?"]
    if not pairs:
      models[g.txt] = None
      continue
    m = len(pairs[0][0])
    A = [[LAMBDA if i == j else 0.0 for j in range(m)] for i in range(m)]
    b = [0.0] * m
    for x, y in pairs:
      for i in range(m):
        b[i] += x[i] * y
        for j in range(m): A[i][j] += x[i] * x[j]
    models[g.txt] = _solve(A, b)
  def predict(row: list) -> dict[str, float]:
    x = feats(row)
    return {name: (sum(w * v for w, v in zip(wt, x)) if wt else 0.0)
            for name, wt in models.items()}
  return predict

def _race_guessers(train_csv, full) -> dict[str, float]:
  "Mean scaled error of each guesser on each goal of every test row."
  train = ezr.Tbl(ezr.csv(train_csv))
  tg    = _goals(train)
  scale = {c.txt: (c.sd or 1.0) for c in tg}
  linear = _linear(train)
  at     = {name: i for i, name in enumerate(full.names)}
  errs   = {"guess_ours": [], "guess_linear": []}
  for row in full.rows:
    inputs = {c.txt: row[c.at] for c in full.x}     # goal cells withheld
    est    = step2.guess(train_csv, inputs)
    lin    = linear(row)
    for g in tg:
      true = row[at[g.txt]]
      if true == "?": continue
      errs["guess_ours"].append(abs(est[g.txt][0] - true) / scale[g.txt])
      errs["guess_linear"].append(abs(lin[g.txt] - true) / scale[g.txt])
  return {k: (sum(v) / len(v) if v else float("inf")) for k, v in errs.items()}

def race(train_csv, test_csv) -> dict[str, float]:
  """Scores for each method on TEST_CSV; lower is better. Tuners: 'ours' (step7),
  'random_search', 'hillclimb' (goal distance of the build picked). Guessers:
  'guess_ours' (step2) and 'guess_linear' (scaled error, trained on TRAIN_CSV)."""
  full  = ezr.Tbl(ezr.csv(test_csv))
  goals = _goals(full)
  if not goals: raise ValueError("the test table has no goal columns")
  if not full.rows: raise ValueError("the test table has no rows")
  rng = random.Random(ezr.the.Seed)
  out = _race_tuners(full, goals, rng)
  out.update(_race_guessers(train_csv, full))
  return out

