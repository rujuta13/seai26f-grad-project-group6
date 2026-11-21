# step12.py
"""Guess a player's goals (penalties, strength, and any others) from the rest.

guess() reuses step2.guess: the nearest training rows vote. The estimate
is their mean, and low..high is their 10th..90th percentile, so the
range is the measure of confidence. A wide range means the neighbours
disagree, and the guess is shaky.
"""
import step2


def guess(train_csv: str, row) -> dict[str, tuple[float, float, float]]:
  """Return {goal: (estimate, low, high)} for every goal of TRAIN_CSV.
  ROW has its goal columns removed: a dict {column name: value}, or the
  input cells in input order. Low..high is the range the true value is
  fairly sure to fall in."""
  out = step2.guess(train_csv, row)
  return {goal: (float(est), float(lo), float(hi))
          for goal, (est, lo, hi) in out.items()}

