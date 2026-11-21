"""Step 28: the smallest set of input changes that brings a customer back to good.

fix(train_csv, row, frozen=()) returns the customer's input columns with the
fewest values changed so that the customer becomes good, using the search in
step18.fix. Columns named in `frozen` are never changed. Goal columns and
ignored 'X' columns are never changed either.

For the client's two questions, the caller sets the arguments:
  - "This customer is drifting away: what's the smallest nudge that brings
    them back?": pass the customer's input columns as `row` (goals removed).
    The changed columns in the result are the nudges to try.
  - "Which product offer would turn a deal-only shopper into a regular one?":
    pass the shopper's input columns as `row` and freeze the columns that
    cannot be offered (for example, demographic or fixed history columns) in
    `frozen`. The changed columns then name the offers to aim for.
"""
from step18 import fix as _fix18


def fix(train_csv, row, frozen=()) -> dict:
    """Inputs of `row` with as few values changed as possible so it becomes good.

    `row` is a customer's input columns (goals removed). Columns named in
    `frozen` (a column name, or an iterable of names) are never changed.
    Returns a new dict with the same keys as `row`. The original row is
    returned if it is already good or if no change helps.
    """
    return _fix18(train_csv, row, frozen=frozen)

