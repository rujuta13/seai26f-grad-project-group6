from step8 import fix as _fix


def fix(train_csv, row, frozen=()) -> dict:
    """Return the same row's inputs with as few values changed as possible so
    that it becomes good. Columns named in `frozen` are never changed.

    `row` is a bad row with its goal columns removed. The result keeps the
    row's own columns, changes only input values, and never adds goal values.
    """
    if isinstance(frozen, str):
        frozen_set = {frozen}
    else:
        frozen_set = set(frozen)

    original = dict(row)
    out = dict(_fix(train_csv, original, frozen=frozen_set))

    # Enforce the guarantees explicitly: frozen columns keep their original
    # values, and no column absent from the input row is introduced.
    for c in frozen_set:
        if c in original:
            out[c] = original[c]
    for c in list(out):
        if c not in original:
            del out[c]

    return out

