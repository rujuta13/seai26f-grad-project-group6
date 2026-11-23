from step8 import fix as _fix


def fix(train_csv, row, frozen=()) -> dict:
    """Return the row's inputs with as few changes as possible so the row becomes good.

    `row` is a bad row with its goals removed. "Good" is the verdict step6.why
    uses: the average standardized goal score is at least the training median.
    Columns named in `frozen` are never changed. Covers both client questions:
    the shortlist fix for a close player, and the training focus that would
    turn a weak penalty-taker into a reliable one.
    """
    return _fix(train_csv, row, tuple(frozen))

