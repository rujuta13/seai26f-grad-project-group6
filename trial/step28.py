"""step28.py - client question 28: fix.

fix(train_csv, row, frozen=()) returns the inputs of a bad row with as few
values changed as possible, so that the row becomes good. Columns named in
`frozen` are never changed.

Two client questions map onto this search:
  - a drifting customer: the changed inputs are the smallest nudge that
    brings their predicted goals back to the good range;
  - a deal-only shopper: the changed inputs are the smallest set of offer
    flags that would turn them into a regular buyer. Freeze any columns
    that cannot be changed, such as demographics, by passing them in `frozen`.

The search is already implemented in step8.fix (candidate changes of 1, 2, ...
input values, scored with step2.guess and the step6 percentile cut-off), so it
is reused here rather than rewritten.
"""
from step8 import fix as _fix

__all__ = ["fix"]


def fix(train_csv, row, frozen=()) -> dict:
    """Inputs of `row` with as few values changed as possible so that the row
    becomes good. `row` is a bad row with goals removed. Columns named in
    `frozen` are never changed. Returns {input column: value} for every input
    column."""
    return dict(_fix(train_csv, row, frozen=frozen))

