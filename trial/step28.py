"""Step 28: client-facing fix for one customer.

fix(train_csv, row, frozen=()): returns the inputs of a customer who is
drifting away, with as few values changed as possible so that the customer is
rated good. Columns named in `frozen` never change.

Note for the client: fix() rates customers on all goals together. The two
requests, a nudge back for a drifting customer and an offer that turns a
deal-only shopper into a regular one, each concern particular goals (for
example NUMDEALSPURCHASES+ and the purchase channel counts). This signature has
no goal parameter, so it cannot target those goals directly. Freezing the
columns you do not want changed (for example, via `frozen`) is the way to
limit the suggested changes to the offer or nudge you have in mind. A
goal-subset version would need its own function with a goal list parameter,
as noted in step11.
"""

import step18


def fix(train_csv, row, frozen=()) -> dict:
    """Return the inputs of `row` with as few values changed as possible so that
    the customer is rated good. Columns named in `frozen` must not change.

    `row` is a bad customer with its goal columns removed. The result has one
    entry per input column of train_csv.
    """
    return dict(step18.fix(train_csv, row, frozen=tuple(frozen)))

