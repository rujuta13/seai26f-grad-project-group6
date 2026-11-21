"""Step 26: justify the campaign group and single customer ratings in plain words.

explain() gives the reasons that make rows good, with the counts behind each
claim so a skeptical expert can recompute them. why() rates one customer row
against the training data and states the two facts that drive the rating.
Both delegate to step6, which holds the implementations.
"""

from step6 import explain as _explain_step6
from step6 import why as _why_step6


def explain(train_csv) -> str:
    """Return plain-language reasons that make rows good, with the counts behind them."""
    return str(_explain_step6(train_csv))


def why(train_csv, row) -> str:
    """Return one or two sentences saying whether this row, goals included, is good or bad, and why."""
    return str(_why_step6(train_csv, dict(row)))

