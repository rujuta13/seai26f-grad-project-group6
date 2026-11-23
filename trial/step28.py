"""step28.py: smallest input nudge that brings a drifting customer back to good.

Both client questions are answered by the same search, so fix() delegates to
step18.fix, which in turn wraps step8.fix:

* "This customer is drifting away: what's the smallest nudge that brings them
  back?" Call fix() with the customer's current inputs as `row`. The result
  changes as few input columns as possible; columns in `frozen` never change.
* "Which product offer would turn a deal-only shopper into a regular one?" Call
  fix() with the shopper's inputs and `frozen` set to every column that cannot
  be changed by an offer. The changed columns are the candidate levers.

Observed builds on the training Pareto front are preferred, and predicted
builds (step2.guess via step8) are used only when they need strictly fewer
changes. Caveat: if no good configuration is found within the search budget,
the result is the lowest-scoring predicted configuration and is not guaranteed
to be good.
"""
from __future__ import annotations

from step18 import fix as _fix


def fix(train_csv, row, frozen=()) -> dict:
    """Inputs of `row` with as few values changed as possible so the row becomes good.

    train_csv: path to the MOOT training csv.
    row: dict {input column name: value} for a bad row, with goal columns removed.
    frozen: input column names that must not change. A single column name given
        as a string is treated as one frozen column.
    Returns a fresh {input column name: value} dict for every input column of the
    training file. The caller's `row` is not modified.
    """
    return _fix(train_csv, dict(row), frozen=frozen)

