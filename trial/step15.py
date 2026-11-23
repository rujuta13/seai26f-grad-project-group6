"""step15.py: blame for bad rows, answering the client's questions about weak spot-kick
players and physically strong players.

The ranking is the one already implemented in step5: input columns are ordered by how
differently their values are distributed among bad rows (bad on the goals) and good rows,
with statistically real differences ranked ahead of noise.
"""

from step5 import blame as _blame


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Columns whose difference between bad and good rows is statistically real
    (step3.same) rank ahead of those that look like noise.
    """
    return list(_blame(train_csv))

