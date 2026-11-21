"""Step 18: fix a bad build with as few input changes as possible.

fix(train_csv, row, frozen=()) returns the inputs of `row` with the fewest
values changed so that the build becomes good, using the search in step8.fix.
Columns named in `frozen` are never changed. Goal columns and ignored 'X'
columns are never returned with a changed value either.

For the client's two questions, the caller sets the arguments:
  - "This player is close: what would he need to improve to make our
    shortlist?": pass his input columns as `row` (goals removed). The
    changed columns in the result are the improvements to aim for.
  - "What training focus would turn our weak penalty-taker into a reliable
    one?": pass his input columns as `row` and freeze the columns that
    cannot be trained (for example, Height or Preferred foot) in `frozen`.
"""
import math

from step1 import _load
from step8 import fix as _fix8


def _is_goal(c):
    return c.endswith("+") or c.endswith("-")


def _is_input(c):
    return not _is_goal(c) and not c.endswith("X")


def _same(a, b):
    """True if two cells hold the same value, comparing floats with a tolerance."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    return a == b


def fix(train_csv, row, frozen=()) -> dict:
    """Inputs of `row` with as few values changed as possible so it becomes good.

    `row` is a bad build's input columns (goals removed). Columns named in
    `frozen` (a column name, or an iterable of names) are never changed.
    Returns a new dict with the same keys as `row`. The original row is
    returned if it is already good or if no change helps.
    """
    if isinstance(frozen, str):
        frozen = (frozen,)
    frozen = tuple(frozen)
    frozen_set = set(frozen)

    base = dict(row)
    header, _ = _load(train_csv)
    allowed = {c for c in header if _is_input(c)}

    result = _fix8(train_csv, base, frozen=frozen)

    out = dict(base)
    for c, v in result.items():
        if c not in base:
            continue
        if c in frozen_set or c not in allowed:
            continue
        if not _same(out[c], v):
            out[c] = v

    # Frozen and non-input columns must keep their original values.
    for c in base:
        if c in frozen_set or c not in allowed:
            out[c] = base[c]

    return out

