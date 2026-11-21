# step20.py
"""Race for the client: does our shortlist beat a top-overall-rating list,
and does our penalty guesser beat an off-the-shelf model?

Shortlist scores (higher is better) use wins: 100 at the best build in
test_csv, 0 at an average one, on goals only. Rivals: step10's random
search and stdlib hill climber, plus "top_overall", which labels the
BUDGET rows with the highest OVR and keeps the best of them.

Penalty scores (higher is better) use skill: 100 when the guesses have
zero mean absolute error, 0 when they are no better than always guessing
the training mean. Rivals: "penalty_ours" is step12.guess, and
"penalty_linear" is ridge regression on the numeric inputs, written with
the standard library since no third-party model library is allowed. Only
the penalty comparison learns from train_csv.
"""
import ezr
import step2
import step4
import step7
import step10
import step12

LAM = 1.0     # ridge penalty for the off-the-shelf linear model


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
  "Solve A x = b by Gaussian elimination with partial pivoting."
  n = len(b)
  m = [a[i][:] + [b[i]] for i in range(n)]
  for c in range(n):
    p = max(range(c, n), key=lambda r: abs(m[r][c]))
    m[c], m[p] = m[p], m[c]
    piv = m[c][c] or 1e-12
    for r in range(c + 1, n):
      f = m[r][c] / piv
      for k in range(c, n + 1): m[r][k] -= f * m[c][k]
  x = [0.0] * n
  for r in range(n - 1, -1, -1):
    s = m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))
    x[r] = s / (m[r][r] or 1e-12)
  return x


def _ridge(tbl, target, lam: float = LAM):
  "Fit ridge regression of TARGET on the numeric inputs of TBL; return row -> guess."
  feats = [c for c in tbl.x if type(c) is ezr.Num]
  stats = []
  for c in feats:
    vs = [r[c.at] for r in tbl.rows if step2._known(r[c.at])]
    mu = sum(vs) / len(vs) if vs else 0.0
    sd = ((sum((v - mu) ** 2 for v in vs) / len(vs)) ** .5) if vs else 0.0
    stats.append((mu, sd or 1.0))

  def z(row) -> list[float]:
    "Standardise a row's numeric inputs; unknown cells sit at the mean (0)."
    return [(row[c.at] - mu) / sd if step2._known(row[c.at]) else 0.0
            for c, (mu, sd) in zip(feats, stats)]

  pairs = [(z(r), r[target.at]) for r in tbl.rows
           if step2._known(r[target.at])]
  ybar  = sum(y for _, y in pairs) / len(pairs)
  p     = len(feats)
  A     = [[lam if i == j else 0.0 for j in range(p)] for i in range(p)]
  b     = [0.0] * p
  for x, y in pairs:
    yc = y - ybar
    for i in range(p):
      b[i] += x[i] * yc
      for j in range(p):
        A[i][j] += x[i] * x[j]
  w = _solve(A, b)
  return lambda row: ybar + sum(wi * zi for wi, zi in zip(w, z(row)))


def _wins(tbl):
  "Scorer: 100 at the best row of TBL, 0 at an average one, on goals only."
  ys = sorted(step4._ydist(tbl, r) for r in tbl.rows)
  lo, avg = ys[0], sum(ys) / len(ys)

  def wins(i: int) -> float:
    y = step4._ydist(tbl, tbl.rows[i])
    return float(max(-100.0, min(100.0,
                                 100 * (1 - (y - lo) / (avg - lo + 1e-32)))))
  return wins


def _ovr_at(tbl) -> int:
  "Column position of the OVR (overall rating) column."
  if "OVR" not in tbl.names:
    raise ValueError(f"race: no OVR column; have {list(tbl.names)}")
  return list(tbl.names).index("OVR")


def _top_overall(tbl, cap: int) -> int:
  "The row a simple rating list picks: the CAP highest OVR rows, best one kept."
  at = _ovr_at(tbl)

  def key(i: int) -> float:
    v = tbl.rows[i][at]
    return -v if step2._known(v) else 1e30

  picks = sorted(range(len(tbl.rows)), key=key)[:cap]
  lab   = ezr.clone(tbl)
  for i in picks: ezr.addRow(lab, tbl.rows[i])
  goals = step4._goals(lab)            # goal stats learnt from the labels only
  return min(picks, key=lambda i: step7._cost(lab, tbl.rows[i], goals))


def _penalty_goal(tbl):
  "The Penalties goal column of TBL."
  hits = [g for g in step4._goals(tbl)
          if g.txt.rstrip("+-").lower() == "penalties"]
  if not hits:
    raise ValueError(f"race: no Penalties goal in "
                     f"{[g.txt for g in step4._goals(tbl)]}")
  return hits[0]


def _skill(truth: list[float], pred: list[float], base: float) -> float:
  "100 at zero mean absolute error, 0 at the error of always guessing BASE."
  n    = len(truth)
  mae  = sum(abs(t - p) for t, p in zip(truth, pred)) / n
  mae0 = sum(abs(t - base) for t in truth) / n
  return float(100 * (1 - mae / (mae0 + 1e-32)))


def _penalty(train_csv: str, test_csv: str) -> dict[str, float]:
  "Penalty guessers, scored by skill on the known penalty cells of test_csv."
  ttbl = step2._load(train_csv)
  tbl  = step2._load(test_csv)
  pen  = _penalty_goal(ttbl)
  goal_names = {g.txt for g in step4._goals(tbl)}

  known = [r[pen.at] for r in ttbl.rows if step2._known(r[pen.at])]
  base  = sum(known) / len(known)     # always guess the training mean

  lin = _ridge(ttbl, pen)
  idx   = [i for i, r in enumerate(tbl.rows) if step2._known(r[pen.at])]
  truth = [tbl.rows[i][pen.at] for i in idx]
  ours, linear = [], []
  for i in idx:
    row = tbl.rows[i]
    q   = {name: row[at] for at, name in enumerate(tbl.names)
           if name not in goal_names}      # inputs only, goals hidden
    ours.append(step12.guess(train_csv, q)[pen.txt][0])
    linear.append(lin(row))
  return {"penalty_ours":   _skill(truth, ours, base),
          "penalty_linear": _skill(truth, linear, base)}


def race(train_csv: str, test_csv: str) -> dict[str, float]:
  "Scores on test_csv for ours and the rivals; higher is better for every key."
  out  = dict(step10.race(train_csv, test_csv))   # ours, random, stdlib_hillclimb
  tbl  = step2._load(test_csv)
  wins = _wins(tbl)
  cap  = min(step10.BUDGET, len(tbl.rows))
  out["top_overall"] = wins(_top_overall(tbl, cap))
  out.update(_penalty(train_csv, test_csv))
  return {k: float(v) for k, v in out.items()}

