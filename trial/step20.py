# step20.py
#!/usr/bin/env python3 -B
"""
step20.py: race.  Our shortlist and our penalty guesser against the rivals
the client named, on test_csv.

  "Does our shortlist beat a simple top overall rating list?"
      ours      step7.choose (ezr.acquire under a label budget).
      top_ovr   the BUDGET rows with the highest OVR, labelled, best one picked.
      random    a random sample of BUDGET rows, best one picked (step10).
      ols       the ridge model's ranking, shortlist of BUDGET (step10).

  "Does our penalty guesser beat an off-the-shelf model?"
      penalty_ours  step2.guess's median estimate of Penalties+.
      penalty_ols   the ridge model's predicted Penalties+ (step10).

Every score is lower-is-better.  Shortlist scores are the true ydist of the
chosen build (0 is ideal).  Penalty scores are the mean |guess - truth| for
Penalties+, in training standard-deviation units.
Reuses step2 (guess), step7 (choose), step10 (race, ridge model) and ezr.
"""
import ezr
import step2, step7, step10

OVR = "OVR"          # the simple rival's rating column (an input)
PEN = "Penalties+"   # the goal the penalty guesser is judged on

def _goal(tbl, name: str):
  "The goal column of TBL named NAME, or raise."
  for c in tbl.y:
    if c.txt == name: return c
  raise ValueError(f"no goal named {name!r} in {[c.txt for c in tbl.y]}")

def _ovr_at(tbl) -> int:
  "Column index of the OVR input, or raise."
  for c in tbl.x:
    if c.txt == OVR: return c.at
  raise ValueError(f"no input column named {OVR!r} in {tbl.names}")

def _top_ovr(full, budget: int) -> list[int]:
  "Indices of the BUDGET rows with the highest OVR: the simple rating list."
  at = _ovr_at(full)
  def rating(i):
    v = full.rows[i][at]
    return v if v != "?" else float("-inf")
  return sorted(range(len(full.rows)), key=rating, reverse=True)[:budget]

def _pick(full, idx: list[int]) -> int:
  "The best of the labelled rows IDX, judged on those labels alone (as step7 does)."
  scorer = ezr.clone(full, [full.rows[i] for i in idx])
  return min(idx, key=lambda i: ezr.ydist(scorer, full.rows[i]))

def _score(full, i: int) -> float:
  "The real score of test row I (all goals known here): lower is better."
  return ezr.ydist(full, full.rows[i])

def _pen_error(train, full, guesses: list[float]) -> float:
  "Mean |guess - truth| for Penalties+, in training sd units."
  truth = _goal(full, PEN)
  sd    = _goal(train, PEN).sd or 1
  errs  = [abs(g - r[truth.at]) / sd
           for r, g in zip(full.rows, guesses) if r[truth.at] != "?"]
  return sum(errs)/len(errs) if errs else 0.0

def race(train_csv: str, test_csv: str) -> dict[str, float]:
  "Shortlist and penalty scores on test_csv, one entry per method; lower is better."
  base   = step10.race(train_csv, test_csv)   # raises on an empty test_csv
  train  = ezr.Tbl(ezr.csv(train_csv))
  full   = ezr.Tbl(ezr.csv(test_csv))
  budget = min(ezr.the.Stop, len(full.rows))

  out = {"ours":   base["ours"],
         "random": base["random"],
         "ols":    base["ols"]}
  out["top_ovr"] = _score(full, _pick(full, _top_ovr(full, budget)))

  names   = full.names
  goal_nm = {names[c.at] for c in full.y}
  inputs  = [{names[at]: v for at, v in enumerate(r) if names[at] not in goal_nm}
             for r in full.rows]

  ours_pen = [step2.guess(train_csv, inp)[PEN][0] for inp in inputs]
  feats, betas = step10._ols(train)
  ols_pen = [step10._predict(feats, betas, r)[PEN] for r in full.rows]

  out["penalty_ours"] = _pen_error(train, full, ours_pen)
  out["penalty_ols"]  = _pen_error(train, full, ols_pen)
  return out

