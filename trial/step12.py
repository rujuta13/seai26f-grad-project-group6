# step12.py
import step2

def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
  """For each goal: (estimate, low, high), from ROW's nearest train neighbours.
  ROW has its goal columns removed. The estimate is the neighbours' mean goal
  value, and low..high is the range where the true value is likely to fall."""
  return step2.guess(train_csv, row)

