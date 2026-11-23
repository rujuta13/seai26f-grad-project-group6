"""step28.py: smallest nudge that brings a drifting customer back, or turns a deal-only
shopper into a regular one.

Both client questions are answered by the minimal-change fix already in step18 (and
step8 underneath). The fix is a measured training build that agrees with the frozen
settings and changes as few other settings as possible, so no new modelling is needed.
"""

from step18 import fix as _fix


def fix(train_csv, row, frozen=()) -> dict:
    """Return the row's inputs with as few values changed as possible so the build is good.

    `row` is a bad row with its goal columns removed. Columns named in `frozen` keep
    their current values. Raises ValueError if `frozen` names a column that is not an
    input, or if no measured build has the frozen settings.
    """
    return dict(_fix(train_csv, row, frozen=frozen))

