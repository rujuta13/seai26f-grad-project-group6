"""Step 18: fix a bad build by changing as few flags as possible.

fix() takes a bad build (flag settings only, goals removed) and returns the same
inputs with as few values changed as the training data predicts are needed to
make it good. The search is delegated to step8.fix, which holds the
implementation. Columns named in `frozen` keep their original values, and the
result contains only input columns.
"""

from step8 import fix as _fix_step8


def fix(train_csv, row, frozen=()) -> dict:
    """Return the row's inputs with as few values changed as possible so it is predicted good.

    `row` is a bad build with goal columns removed. Columns named in `frozen`
    are never changed. If the row is already predicted good, it is returned
    unchanged.
    """
    if isinstance(frozen, str):
        frozen = (frozen,)
    frozen = tuple(frozen)

    result = dict(_fix_step8(train_csv, dict(row), frozen=frozen))

    # Safety net: frozen columns must keep their original values.
    for c in frozen:
        if c in row and c in result:
            result[c] = row[c]
    return result

