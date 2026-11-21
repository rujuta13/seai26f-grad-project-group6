"""Step 28: fix. The smallest nudge that brings a drifting or weak row back to good.

Serves the client's questions:
  - "This customer is drifting away: what's the smallest nudge that brings them back?"
  - "Which product offer would turn a deal-only shopper into a regular one?"

Thin wrapper around step8.fix. The search tries changes to one input column,
then two, and so on, and rates each candidate with step2.guess and the scoring
of step6.why. Goal columns passed in `row` are ignored. Names in `frozen` must
be input columns, since a misspelt name would otherwise leave a column free to
change without warning.
"""

import step8


def fix(train_csv, row, frozen=()) -> dict:
    """Same inputs as `row`, with as few values changed as possible so it becomes good.

    row: a bad row with goal columns removed (goal cells are ignored if present).
    frozen: input column names that must not change. A single name given as a
        string is also accepted.
    Returns {input: value} for the input columns of `row`. A row that is already
    good is returned unchanged. If no good row is found within the search budget,
    the best-rated change set found is returned.
    Raises ValueError if `frozen` names a column that is not an input.
    """
    inputs, _, _, _ = step8._model(train_csv)
    if isinstance(frozen, str):
        frozen = (frozen,)
    frozen = tuple(frozen)

    unknown = [name for name in frozen if name not in inputs]
    if unknown:
        raise ValueError(f"frozen names are not input columns: {unknown}")

    return step8.fix(train_csv, dict(row), frozen=frozen)

