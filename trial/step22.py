# step22.py
import step12

def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
  """For each goal: (estimate, low, high). ROW has its goal columns removed.
  The goals include NUMWEBPURCHASES+ (online) and NUMSTOREPURCHASES+ (in store),
  and RESPONSE+ (whether the customer answers the next campaign). low..high is
  the range where the true value is likely to fall."""
  return step12.guess(train_csv, row)

