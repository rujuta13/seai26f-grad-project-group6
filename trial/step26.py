"""Step 26: justify. Two client questions, answered in plain words.

Standard library only. Both answers come from step 16: explain() gives the
reasons the recommended group is good, and why() rates one row in one or two
sentences. This module adds one check: a row missing an input column is not
rated, because step 2 would silently fill that column with 0, and the reasons
given would then be wrong.
"""

from step2 import _model
from step16 import explain as _explain, why as _why


def explain(train_csv) -> str:
    """Plain-words reasons, for the sales director, for what makes the
    recommended group good and which settings are tied to the weakest builds."""
    return _explain(train_csv)


def why(train_csv, row) -> str:
    """One or two sentences on whether this row (goals included) is good or
    bad, and why. A row missing an input column is not rated."""
    model = _model(train_csv)
    missing = [c for c in model["inputs"] if c not in row]
    if missing:
        return (
            "Not rated: the row is missing input columns ("
            + ", ".join(missing)
            + "), so its reasons cannot be checked against the training data."
        )
    return _why(train_csv, row)

