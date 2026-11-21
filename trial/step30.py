# step30.py
#!/usr/bin/env python3 -B
"""
step30.py: race.  Our targeting and our response guesser against the rivals
the client named, on test_csv.

  "Does our targeting beat simply mailing our highest-income customers?"
      ours        step7.choose (ezr.acquire under a label budget); score is
                  the true ydist of the chosen customer (0 is ideal).
      income_top  the BUDGET customers with the highest INCOME, labelled, and
                  the best one picked by the same rule as step7.
      random      a random sample of BUDGET customers, best one picked.

  "Does our response guesser beat an off-the-shelf model?"
      response_ours  step2.guess's median estimate of RESPONSE+.
      response_ols   the ridge model's predicted RESPONSE+ (step10).

Every score is lower-is-better.  Targeting scores are the true ydist of the
chosen customer (0 is ideal).  Response scores are the mean |guess - truth|
for RESPONSE+, in training standard-deviation units.
Reuses step2 (guess), step7 (choose), step10 (blind writer, ridge model) and
ezr for tables and distance.
"""
import os, random, tempfile
import ezr
import step2, step7, step10

INCOME = "INCOME"      # the simple rival's rating column (an input)
RESP   = "RESPONSE+"   # the goal the response guesser is judged on

def _at(tbl, name: str) -> int:
  "Column index of the column named NAME in TBL, or raise."
  if name not in tbl.names:
    raise ValueError(f"no column named {name!r} in {tbl.names}")
  return tbl.names.index(name)

def _resp_col(tbl):
  "The RESPONSE+ goal column of TBL, or raise."
  for c in tbl.y:
    if c.txt == RESP: return c
  raise ValueError(f"no goal named {RESP!r} in {[c.txt for c in tbl.y]}")

def _top_income(full, budget: int) -> list[int]:
  "Indices of the BUDGET rows with the highest INCOME: the simple rival."
  at = _at(full, INCOME)
  def income(i):
    v = full.rows[i][at]
    return v if v != "?" else float("-inf")
  return sorted(range(len(full.rows)), key=income, reverse=True)[:budget]

def _pick(full, idx: list[int]) -> int:
  "The best of the labelled rows IDX, judged on those labels alone (as step7 does)."
  scorer = ezr.clone(full, [full.rows[i] for i in idx])
  return min(idx, key=lambda i: ezr.ydist(scorer, full.rows[i]))

def _score(full, i: int) -> float:
  "The real ydist of test row I (all goals known here): lower is better."
  return ezr.ydist(full, full.rows[i])

def _resp_error(train, full, guesses: list[float]) -> float:
  "Mean |guess - truth| for RESPONSE+, in training sd units."
  truth = _resp_col(full).at
  sd    = _resp_col(train).sd or 1
  errs  = [abs(g - r[truth]) / sd
           for r, g in zip(full.rows, guesses) if r[truth] != "?"]
  return sum(errs) / len(errs) if errs else 0.0

def race(train_csv: str, test_csv: str) -> dict[str, float]:
  "Targeting and response scores on test_csv, one entry per method; lower is better."
  random.seed(ezr.the.Seed)
  train   = ezr.Tbl(ezr.csv(train_csv))
  full    = ezr.Tbl(ezr.csv(test_csv))
  n       = len(full.rows)
  if n == 0: raise ValueError("no rows in test_csv")
  names   = full.names
  goal_at = [c.at for c in full.y]
  goal_nm = {names[at] for at in goal_at}
  budget  = min(ezr.the.Stop, n)

  def label(i) -> dict:
    "Reveal test row I's goals, in the form step7's oracle expects."
    return {names[at]: full.rows[i][at] for at in goal_at}

  out = {}

  # ours: ezr.acquire under a label budget, via step7.choose
  with tempfile.TemporaryDirectory() as d:
    blind = os.path.join(d, "blind.csv")
    step10._write_blind(full, goal_at, blind)
    out["ours"] = _score(full, step7.choose(blind, label, budget))

  # rivals for targeting
  out["income_top"] = _score(full, _pick(full, _top_income(full, budget)))
  out["random"]     = _score(full, _pick(full, random.sample(range(n), budget)))

  # response guesser: ours versus the ridge model
  inputs = [{names[at]: v for at, v in enumerate(r) if names[at] not in goal_nm}
            for r in full.rows]
  ours_resp = [step2.guess(train_csv, inp)[RESP][0] for inp in inputs]
  feats, betas = step10._ols(train)
  ols_resp = [step10._predict(feats, betas, r)[RESP] for r in full.rows]

  out["response_ours"] = _resp_error(train, full, ours_resp)
  out["response_ols"]  = _resp_error(train, full, ols_resp)
  return out

