"""Step 18: fix. The fewest input changes that make a bad flag set good.

Standard library only. The search, the good/bad rule (predicted 0-to-1 score
at or above the training median) and the handling of frozen columns are
reused from step8.fix. This module adds argument checks and a guarantee the
client needs: frozen columns come back unchanged, and goal cells are never
read or returned.
"""

from step2 import _model
from step8 import fix as _fix


def fix(train_csv, row, frozen=()) -> dict:
    """Inputs of `row` with as few values changed as possible so the build is
    predicted good. Columns named in `frozen` keep their values.

    `row` holds the input columns (goal columns are ignored). Returns a dict
    of input column names to values. Raises ValueError if a name in `frozen`
    is not an input column of train_csv, or if a frozen column is absent
    from `row`.
    """
    model = _model(train_csv)
    inputs = model["inputs"]

    if isinstance(frozen, str):
        frozen = (frozen,)
    frozen = tuple(frozen)

    unknown = [c for c in frozen if c not in inputs]
    if unknown:
        raise ValueError(
            "frozen names are not input columns: " + ", ".join(map(str, unknown))
        )
    absent = [c for c in frozen if c not in row]
    if absent:
        raise ValueError(
            "frozen columns are missing from row: " + ", ".join(map(str, absent))
        )

    base = {c: row[c] if c in row else 0.0 for c in inputs}
    result = dict(_fix(train_csv, row, frozen=frozen))

    for c in frozen:
        if result[c] != base[c]:
            raise RuntimeError(f"frozen column {c!r} was changed")

    return {c: result[c] for c in inputs}

