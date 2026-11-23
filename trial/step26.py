"""step26.py: justify the campaign group and single-customer ratings in plain words.

Both client questions are answered by the explanations already in step16 (and step6 underneath):
- explain: reasons that make rows good, checkable by a skeptic.
- why: one or two sentences on why a given row is rated good or bad.
"""

from step16 import explain as _explain
from step16 import why as _why


def explain(train_csv) -> str:
    """Plain-words reasons for what makes rows good, checkable by a skeptic."""
    return str(_explain(train_csv))


def why(train_csv, row) -> str:
    """One or two sentences on why this row (goals included) is good or bad."""
    return str(_why(train_csv, row))

