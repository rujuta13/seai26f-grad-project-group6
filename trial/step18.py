"""step18.py: smallest change to a bad build that makes it good, leaving frozen settings alone.

The work is done by step8.fix, which picks a measured training build that agrees
with the frozen settings and changes as few other settings as possible. A single
column name passed as `frozen` is treated as one column, not as a sequence of characters.
"""

from step8 import fix as _fix


def fix(train_csv, row, frozen=()) -> dict:
    """Return the bad row's inputs with as few values changed as possible so the build is good.

    `row` is a bad row with its goal columns removed. Columns named in `frozen` keep
    their current values. Raises ValueError if `frozen` names a column that is not an
    input, or if no measured build has the frozen settings.
    """
    if isinstance(frozen, str):
        frozen = (frozen,)
    return _fix(train_csv, row, frozen=tuple(frozen))

