"""step18.py: fix, for the two client questions.

Q1: "This player is close: what would he need to improve to make our shortlist?"
Q2: "What training focus would turn our weak penalty-taker into a reliable one?"

Both questions ask for the fewest input values to change so that a bad row
becomes good. That is step8.fix: a beam search over single-value changes, with
goals guessed from the nearest training rows and rated with the same score and
top-quarter cutoff as step6.why. Columns named in `frozen`, such as age or
preferred foot, never change. This module reuses step8.fix and only fixes the
frozen argument to a tuple of names before delegating.
"""
from __future__ import annotations

from step8 import fix as _fix


def fix(train_csv, row, frozen=()) -> dict:
    """Input values for the bad row `row`, with as few values changed as possible
    so that it is rated good. Returns a dict over every input column.

    row: a row's input cells. Goal cells are ignored, and missing cells are
         filled with training defaults.
    frozen: input column names that must not change. A single name or an
            iterable of names is accepted. Names that are not input columns
            raise ValueError.
    """
    if isinstance(frozen, str):
        frozen = (frozen,)
    else:
        frozen = tuple(frozen)
    return dict(_fix(train_csv, row, frozen=frozen))

