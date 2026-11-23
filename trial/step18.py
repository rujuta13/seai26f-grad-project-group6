"""Step 18: client-facing fix for a single build.

fix(train_csv, row, frozen=()): returns the inputs of a bad build with as few
values changed as possible so that it is rated good. Columns named in `frozen`
never change.

Note for the client: fix() rates builds on all goals together. The penalty-taker
request concerns one goal (Penalties+), and this signature has no goal
parameter, so it can only partly answer that question. A goal-subset version
would need its own function with a goal list parameter, as noted in step11.
"""

import csv

import step8


def _input_columns(train_csv) -> list[str]:
    with open(train_csv, newline="") as f:
        cols = list(csv.DictReader(f).fieldnames or [])
    return [c for c in cols if not c.endswith(("+", "-", "X"))]


def fix(train_csv, row, frozen=()) -> dict:
    """Return the inputs of `row` with as few values changed as possible so that
    the build is rated good. Columns named in `frozen` must not change.

    `row` is a bad build with its goal columns removed. The result has one entry
    per input column of train_csv.
    """
    frozen = tuple(frozen)
    inputs = _input_columns(train_csv)

    proposed = step8.fix(train_csv, row, frozen=frozen)

    out = {}
    for c in inputs:
        if c in frozen:
            out[c] = row.get(c)
        elif c in proposed:
            out[c] = proposed[c]
        else:
            out[c] = row.get(c)
    return out

