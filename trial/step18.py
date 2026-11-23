"""Step 18: client-facing wrapper around step8.fix for improving a near-miss build."""

from step1 import _read, _columns
from step8 import fix as _fix


def fix(train_csv, row, frozen=()) -> dict:
    """Return the row's inputs changed as little as possible so the build is good.

    `row` is a bad row with its goal values removed; any goal keys present are
    ignored. The result contains only input columns, with each value either
    unchanged from `row` or taken from an observed training build that is good
    by the step6 criterion. Columns named in `frozen` are never changed. Among
    admissible candidates, the one differing from `row` in the fewest inputs is
    chosen, with ties going to the higher average percentile and then to the
    lower training index. If no admissible good build exists, the row's inputs
    are returned unchanged.
    """
    fieldnames, _rows = _read(train_csv)
    inputs, _goals = _columns(fieldnames)

    frozen_names = tuple(str(c) for c in frozen)
    result = _fix(train_csv, row, frozen=frozen_names)

    # Only input columns are reported, and frozen columns keep their values.
    out = {}
    for c in inputs:
        if c in frozen_names:
            out[c] = row.get(c)
        else:
            out[c] = result.get(c, row.get(c))
    return out

