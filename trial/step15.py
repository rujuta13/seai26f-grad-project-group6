# step15.py
"""
step15: blame, for the client.  Two asks: why are some players weak from
the spot (Penalties+), and what separates the physically strong players
(Strength+) from the rest?  Both goals count in ydist, so bad rows are
those far from ideal on Penalties and Strength together.  Input columns
are ranked by how concentrated bad rows are on one side of their splits.
This reuses step5's blame unchanged.
"""
import step5


def blame(train_csv: str) -> list[str]:
  """Input column names, most to blame for bad rows (bad on the goals)
  first. Ties keep the order of the header."""
  return list(step5.blame(train_csv))

