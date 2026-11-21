"""Step 24: flow. Welcome-offer triage for new sign-ups, and drift warnings for each month's sign-ups.

Serves the client's requests:
  - "As new customers sign up, tell me straight away who's worth a welcome offer."
  - "Warn me when this month's sign-ups stop looking like our usual customers."

Thin wrappers around step14, which already applies both rules causally:
stream() keeps a sign-up whose normalised goal total reaches the 95th percentile
of the training totals. The decision uses only train_csv and that row, never
later rows. drift() checks every few sign-ups, compares the most recent window
with the training customers one column at a time, and warns once several
consecutive checks find a difference.
"""

import step14


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) or not (False) each arriving row, in order.

    Each decision depends only on train_csv and the row itself, so no later
    sign-up can change an earlier decision.
    """
    return [bool(k) for k in step14.stream(train_csv, rows)]


def drift(train_csv, rows) -> int | None:
    """Index into rows of the first row at which the sign-ups stop looking like
    the training customers, or None if they never do.

    Only rows up to and including the warning index are used to decide it.
    """
    result = step14.drift(train_csv, rows)
    return None if result is None else int(result)

