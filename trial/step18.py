"""Step 18: fix. The fewest input changes that make a bad player good, for the client.

Serves the client's requests:
  - "This player is close: what would he need to improve to make our shortlist?"
  - "What training focus would turn our weak penalty-taker into a reliable one?"

Method: step8.fix does the search. It tries changes to one input column, then
two, and so on, and rates each candidate with step2.guess and the same scoring
as step6.why. This module adapts that search to the client's wording: goal
columns passed in `row` are ignored, `frozen` must name real input columns,
and a single frozen name given as a string is accepted.
"""

import step8


def fix(train_csv, row, frozen=()) -> dict:
    """Same inputs as `row`, with as few values changed as possible so it becomes good.

    row: a bad player with goal columns removed (goal cells are ignored if present).
    frozen: input column names that must not change, such as attributes a
        training plan cannot alter. A single name given as a string is also accepted.
    Returns {input: value} for the input columns of `row`. A player who is
    already good is returned unchanged. If no good player is found within the
    search budget, the best-rated change set found is returned.
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

