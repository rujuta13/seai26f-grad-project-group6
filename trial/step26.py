"""step26.py - client question 26: justify.

explain(train_csv) gives plain-words reasons, checkable by a skeptical expert,
for what makes customers good. It can be used to explain to the sales director
why a group should get the next campaign.

why(train_csv, row) says in one or two sentences why a single customer (goals
included) is rated good or bad, with reasons the client can act on when a
customer is rated low value.

Both behaviours are already implemented in step6 (explain and why), so they
are reused here rather than rewritten.
"""
from step6 import explain as _explain
from step6 import why as _why

__all__ = ["explain", "why"]


def explain(train_csv) -> str:
    """Plain-words reasons for what makes rows good, checkable by a skeptic."""
    return str(_explain(train_csv))


def why(train_csv, row) -> str:
    """One or two sentences on why this row (goals included) is good or bad."""
    return str(_why(train_csv, row))

