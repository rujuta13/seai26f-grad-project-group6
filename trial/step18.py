"""step18.py: smallest set of input changes that turns a bad build into a good one.

Both client questions are answered by step8.fix, which searches for the fewest
changed input columns. Observed builds on the training Pareto front are preferred,
and predicted builds (step2.guess) are used only when they need strictly fewer
changes. Columns in `frozen` never change.

This module adds only input handling: a single column name may be passed as a
string, and the result is returned as a fresh dict.

Caveat: if no good build is found within the search budget, step8 returns its
lowest-scoring predicted configuration, which is not guaranteed to be good.
"""
from __future__ import annotations

from step8 import fix as _fix


def fix(train_csv, row, frozen=()) -> dict:
    """Inputs of `row` with as few values changed as possible so the build becomes good.

    row: dict {input column name: value} for a bad build (goals removed).
    frozen: input column names that must not change. A single column name given
        as a string is treated as one frozen column.
    Returns {input column name: value} for every input column of the training file.
    """
    if isinstance(frozen, str):
        frozen = (frozen,)
    return dict(_fix(train_csv, row, frozen=tuple(frozen)))

