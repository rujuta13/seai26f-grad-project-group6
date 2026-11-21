"""step18.py: fix a bad build with as few input changes as possible.

Standard library only (Python 3.13). Reuses step8.fix, which already does
this: it searches candidate builds in order of how many inputs they change,
judges each by the kNN goal estimate from step2 against the training median
(the threshold used by step6 and step8), and returns the smallest change
predicted to be good. Columns named in `frozen` are never changed.

Both client questions use this function. For "what would he need to improve
to make our shortlist", pass his row as `row`. For "what training focus would
make our weak penalty-taker reliable", pass the same row and let the changed
inputs show the training focus. Freeze any input the club cannot change, such
as Age or Nation.
"""
from __future__ import annotations

from step8 import fix as _step8_fix


def fix(train_csv, row, frozen=()) -> dict:
    """Same inputs as `row`, with as few values changed as possible so that
    the build becomes good. Columns named in `frozen` are never changed.

    `row` is a dict {input column name: value} with the goal columns removed.
    Returns a dict with the same keys as `row`, where changed cells hold
    values seen in the training file. If the row is already predicted good,
    it is returned unchanged.
    """
    return _step8_fix(train_csv, row, frozen=frozen)

