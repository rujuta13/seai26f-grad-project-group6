# step10.py
"""Race: our tuner against a random search and a stdlib-only hill climber.

The standard library ships no hyperparameter tuner and no predictive model,
so the stdlib rival here is a plain one-flip hill climber written with the
stdlib alone. It is a stand-in for a library tuner, and it is named as one.

Every method gets the same budget of labelled builds from test_csv. Each
method picks one build. Its pick is scored by wins: 100 at the best build
in test_csv, 0 at an average one, using goals only as step4 does.

train_csv is accepted for the shared signature. The tuners learn only from
the labels they buy on test_csv, so the training split is not used here.
"""
import os
import random
import tempfile

import ezr
import step2
import step4
import step7

BUDGET = 30   # labelled builds per method, the same for every rival


def race(train_csv: str, test_csv: str) -> dict[str, float]:
  "Wins score of the build each method picks from TEST_CSV; higher is better."
  _ = train_csv                        # shared signature; see docstring

  tbl   = step2._load(test_csv)
  rows  = tbl.rows
  n     = len(rows)
  cap   = min(BUDGET, n)
  goals = step4._goals(tbl)
  gat   = {g.at for g in goals}
  truth = [{g.txt: r[g.at] for g in goals} for r in rows]

  def label(i: int) -> dict:
    "Reveal the goals of test build I, as a paid label."
    return truth[i]

  # ---- scoring: wins over goals only, 100 best .. 0 average ----
  ys      = sorted(step4._ydist(tbl, r) for r in rows)
  lo, avg = ys[0], sum(ys) / n

  def wins(i: int) -> float:
    y = step4._ydist(tbl, rows[i])
    return max(-100.0, min(100.0,
                           100 * (1 - (y - lo) / (avg - lo + 1e-32))))

  rng = random.Random(ezr.the.Seed)

  # ---- ours: step7.choose on a blind copy of test_csv ----
  # step7 reads its input from a path, so write a blind copy to a temp file:
  # same header, every goal cell set to '?'.
  text = [",".join(str(s) for s in tbl.names)]
  for r in rows:
    text.append(",".join("?" if at in gat else str(v)
                         for at, v in enumerate(r)))
  fd, path = tempfile.mkstemp(suffix=".csv")
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
      f.write("\n".join(text) + "\n")
    ours = step7.choose(path, label, budget=cap)
  finally:
    os.remove(path)

  # ---- random search: the same budget, picked at random, best kept ----
  def best_of(picks: list[int]) -> int:
    "Label PICKS, then return the one whose goals cost least."
    lab  = ezr.clone(tbl)
    seen = {}
    for i in picks:
      seen[i] = step7._fill(tbl, rows[i], label(i))
      ezr.addRow(lab, seen[i])
    return min(seen, key=lambda i: step7._cost(lab, seen[i], goals))

  rand_pick = best_of(rng.sample(range(n), cap))

  # ---- stdlib-only stand-in tuner: one-flip hill climb ----
  # Start from a few random labels. Each step labels the unlabelled build
  # nearest the current best in x-space, so the search walks locally.
  def hill(first: list[int]) -> int:
    "Hill climb from the labels in FIRST, spending the whole budget."
    lab  = ezr.clone(tbl)
    seen = {}

    def ask(i: int) -> None:
      seen[i] = step7._fill(tbl, rows[i], label(i))
      ezr.addRow(lab, seen[i])

    for i in first:
      ask(i)
    while len(seen) < cap:
      top  = min(seen, key=lambda i: step7._cost(lab, seen[i], goals))
      rest = [i for i in range(n) if i not in seen]
      ask(min(rest, key=lambda i: ezr.xdist(tbl, rows[i], rows[top])))
    return min(seen, key=lambda i: step7._cost(lab, seen[i], goals))

  hill_pick = hill(rng.sample(range(n), min(step7.START, cap)))

  return {
    "ours":             wins(ours),
    "random":           wins(rand_pick),
    "stdlib_hillclimb": wins(hill_pick),
  }

