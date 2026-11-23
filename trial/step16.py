"""step16.py: justify the recommended player and single-player ratings in plain words.

Both client questions are answered by the explanations already in step6:
- explain: reasons that make rows good, checkable by a skeptic.
- why: one or two sentences on why a given row is rated good or bad.
"""

from step6 import explain as _explain
from step6 import why as _why


def explain(train_csv) -> str:
    """Plain-words reasons for what makes players good, checkable by a skeptic."""
    return _explain(train_csv)


def why(train_csv, row) -> str:
    """One or two sentences on why this player (goals included) is good or bad."""
    return _why(train_csv, row)

