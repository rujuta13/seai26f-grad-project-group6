from step8 import fix as _fix


def fix(train_csv, row, frozen=()) -> dict:
    """Return the same build's inputs with as few values changed as possible so
    that it is predicted to be good. Columns named in `frozen` are never changed.

    `row` is a bad build with its goal columns removed. The result keeps the
    row's own columns, changes only input values, and never adds goal values.
    """
    if isinstance(frozen, str):
        frozen_set = {frozen}
    else:
        frozen_set = set(frozen)

    original = dict(row)
    out = dict(_fix(train_csv, original, frozen=frozen_set))

    # Enforce the frozen-column guarantee explicitly, in case the underlying
    # search ever returns a changed value for a protected column.
    for c in frozen_set:
        if c in original:
            out[c] = original[c]

    return out

