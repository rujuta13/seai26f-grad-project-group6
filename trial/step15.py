"""Step 15: which input columns are to blame for bad rows, for the client's two questions.

blame() ranks every input (flag) column by how differently its values are
distributed among bad rows and good rows, where bad rows are the worst quarter
by the normalised goal sum. It delegates to step5, which holds the
implementation. The ranking is over all goals together, so the top columns
explain what sets apart the weakest builds on the goals as a whole, including
'Penalties+' and 'Strength+'.
"""

from step5 import blame as _blame_step5


def blame(train_csv) -> list[str]:
    """Return input column names, most to blame for bad rows (bad on the goals) first."""
    return [str(c) for c in _blame_step5(train_csv)]

