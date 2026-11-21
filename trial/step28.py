"""Step 28: fix. The fewest input changes that bring a bad row back to good.

Standard library only. Both client questions are answered by step 18's fix():
the search, the good/bad rule (predicted 0-to-1 score at or above the training
median) and the checks on frozen columns are reused unchanged. Columns named
in `frozen` keep their values, and goal cells are never read or returned.
"""

from step18 import fix as _fix


def fix(train_csv, row, frozen=()) -> dict:
    """Inputs of `row` with as few values changed as possible so the row is
    predicted good. Columns named in `frozen` keep their values.

    `row` holds the input columns (goal columns are ignored). Returns a dict
    of input column names to values. Raises ValueError if a name in `frozen`
    is not an input column of train_csv, or if a frozen column is absent
    from `row`.
    """
    return _fix(train_csv, row, frozen=frozen)

