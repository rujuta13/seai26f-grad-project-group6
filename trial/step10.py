# step10.py
"""
step10: race.  Our tuner against random search, and our guesser against an
off-the-shelf model, on the same test rows and the same yardstick.

Each method picks ONE row of TEST_CSV as its best flag set. The score is
ezr.wins of that row, using the test rows' own goals: 100 at the best test
row, 0 at an average one, higher is better.

Methods:
  ours        our tuner: step7.choose, labelling at most the.Stop rows.
  random      random search: label the same budget at random, keep the best.
  ours_guess  our guesser: step4's predicted ydist (step2.guess fills the
              goals), ranking every test row with no labels at all.
  knn         off-the-shelf stand-in: k-nearest-neighbour regression on the
              training rows, written with the stdlib only.

The standard library ships no tuner, so the tuner's rival is random search.
No third-party model is used, since only the stdlib and ezr are allowed.

Reuses step2 (model), step4 (_ydist), step7 (choose), and ezr (Tbl, csv,
clone, ydist, xdist, wins, the).
"""
import heapq
import os
import random
import tempfile
import ezr
import step2
import step4
import step7

KNN_K = 5     # neighbours the knn stand-in averages


def _write_blind(tbl, goals) -> str:
  "Write TBL to a temp csv with every goal cell '?'; return its path."
  fd, path = tempfile.mkstemp(suffix=".csv")
  with os.fdopen(fd, "w", encoding="utf-8") as f:
    f.write(",".join(str(n) for n in tbl.names) + "\n")
    for r in tbl.rows:
      cells = ["?" if at in goals else v for at, v in enumerate(r)]
      f.write(",".join(str(v) for v in cells) + "\n")
  return path


def _blind(row, goals) -> list:
  "ROW with its goal cells replaced by '?'."
  return ["?" if at in goals else v for at, v in enumerate(row)]


def _label(tbl, goals):
  "An oracle for step7: row index -> {goal name: true value}."
  def label(i):
    r = tbl.rows[i]
    return {tbl.names[at]: r[at] for at in goals}
  return label


def _ours(path, tbl, goals, budget) -> int:
  "Our tuner: step7.choose on the blind csv. Returns a test row index."
  return step7.choose(path, _label(tbl, goals), budget=budget)


def _random(tbl, budget, seed) -> int:
  "Random search: label BUDGET random rows, keep the best of them."
  rng = random.Random(seed)
  idx = rng.sample(range(len(tbl.rows)), min(budget, len(tbl.rows)))
  lab = ezr.clone(tbl, [tbl.rows[i] for i in idx])   # stats from labelled rows
  return min(idx, key=lambda i: ezr.ydist(lab, tbl.rows[i]))


def _ours_guess(train_csv, tbl, goals) -> int:
  "Our guesser: rank every test row by its predicted ydist. No labels."
  blinds = [_blind(r, goals) for r in tbl.rows]
  preds  = [step4._ydist(train_csv, b) for b in blinds]
  return min(range(len(preds)), key=lambda i: preds[i])


def _knn(train_tbl, tbl, goals) -> int:
  "Off-the-shelf stand-in: kNN regression of each goal, then rank by ydist."
  def predict(row) -> float:
    near = heapq.nsmallest(KNN_K, train_tbl.rows,
                           key=lambda t: ezr.xdist(train_tbl, row, t))
    filled = list(row)
    for at in goals:
      vs = [t[at] for t in near if t[at] != "?"]
      filled[at] = sum(vs) / len(vs) if vs else train_tbl.cols[at].mu
    return ezr.ydist(train_tbl, filled)
  preds = [predict(_blind(r, goals)) for r in tbl.rows]
  return min(range(len(preds)), key=lambda i: preds[i])


def race(train_csv, test_csv) -> dict[str, float]:
  "Each method's pick on TEST_CSV, scored by ezr.wins. Higher is better."
  test  = ezr.Tbl(ezr.csv(test_csv))
  goals = {c.at for c in test.y}
  if not goals:
    raise ValueError("race: test_csv has no goal columns")
  if not test.rows:
    raise ValueError("race: test_csv has no rows")

  budget = min(ezr.the.Stop, len(test.rows))
  score  = ezr.wins(test)
  train_tbl, _ = step2._model(train_csv)

  path = _write_blind(test, goals)
  try:
    picks = {
      "ours":       _ours(path, test, goals, budget),
      "random":     _random(test, budget, ezr.the.Seed),
      "ours_guess": _ours_guess(train_csv, test, goals),
      "knn":        _knn(train_tbl, test, goals),
    }
  finally:
    os.remove(path)

  return {m: float(score(test.rows[i])) for m, i in picks.items()}

