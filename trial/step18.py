import step8


def fix(train_csv, row, frozen=()) -> dict:
    """Inputs of `row`, changed in as few values as possible so it becomes good.

    `row` is a bad row with its goal columns removed. The result has the same
    input columns as `row`. Values are copied from a training row that no
    other training row dominates, chosen by step8.fix. Columns named in
    `frozen` are never changed: they are restored to the values in `row`
    after the search. Names in `frozen` that are not inputs are ignored.

    If no undominated training row agrees with the frozen columns, the result
    may still be bad. If no training row agrees with them, the inputs are
    returned unchanged.
    """
    if isinstance(frozen, str):
        frozen = (frozen,)
    frozen = set(frozen)

    out = step8.fix(train_csv, row, frozen=frozen)

    # guarantee: frozen input columns keep the value they had in `row`
    for c in frozen:
        if c in out:
            out[c] = row.get(c, "")

    return out

