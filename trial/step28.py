"""step28.py: smallest nudge that brings a drifting customer back, and the
product offer that turns a deal-only shopper into a regular one.

Standard library only (Python 3.13). Reuses step18.fix, which already does
this: it searches candidate builds in order of how many inputs they change,
judges each by the kNN goal estimate from step2 against the training median,
and returns the smallest change predicted to be good. Columns named in
`frozen` are never changed.

For "the smallest nudge that brings them back", pass the drifting customer's
row as `row` and freeze inputs the team cannot change, such as AGE or
EDUCATION. For "which product offer would turn a deal-only shopper into a
regular one", the changed inputs show the offer, for example the MNT* spend
columns or ACCEPTEDCMP*, and the same freezing applies.
"""
from __future__ import annotations

from step18 import fix as _step18_fix


def fix(train_csv, row, frozen=()) -> dict:
    """Same inputs as `row`, with as few values changed as possible so that
    the customer becomes good. Columns named in `frozen` are never changed.

    `row` is a dict {input column name: value} with the goal columns removed.
    Returns a dict with the same keys as `row`, where changed cells hold
    values seen in the training file. If the row is already predicted good,
    it is returned unchanged.
    """
    return _step18_fix(train_csv, row, frozen=frozen)

