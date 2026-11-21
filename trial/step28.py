"""step28.py: fix, for the two client questions.

Q1: "This customer is drifting away: what's the smallest nudge that brings them back?"
Q2: "Which product offer would turn a deal-only shopper into a regular one?"

Both questions ask for the fewest input values to change so that a bad row
becomes good. That is step18.fix, which delegates to step8.fix: a beam search
over single-value changes, with goals guessed from the nearest training rows and
rated with the same score and top-quarter cutoff as step6.why. Columns named in
`frozen` never change. Nothing new is computed here, so this module only fixes
the argument and return shape.
"""
from __future__ import annotations

from step18 import fix as _fix


def fix(train_csv, row, frozen=()) -> dict:
    """Input values for the bad row `row`, with as few values changed as possible
    so that it is rated good. Returns a dict over every input column.

    row: a row's input cells. Goal cells are ignored, and missing cells are
         filled with training defaults.
    frozen: input column names that must not change. A single name or an
            iterable of names is accepted. Names that are not input columns
            raise ValueError.
    """
    return dict(_fix(train_csv, row, frozen=frozen))

