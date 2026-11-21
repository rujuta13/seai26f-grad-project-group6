"""step18.py - client question 18: fix.

fix(train_csv, row, frozen=()) returns the inputs of a bad row with as few
values changed as possible, so that the row becomes good. Columns named in
`frozen` are never changed.

The search is already implemented in step8.fix (candidate changes of 1, 2, ...
input values, scored with step2.guess and the step6 percentile cut-off), so it
is reused here rather than rewritten.
"""
from step8 import fix as _fix

__all__ = ["fix"]


def fix(train_csv, row, frozen=()) -> dict:
    """Inputs of `row` with as few values changed as possible so that the row
    becomes good. Columns named in `frozen` are never changed. Returns
    {input column: value} for every input column."""
    return dict(_fix(train_csv, row, frozen=frozen))

