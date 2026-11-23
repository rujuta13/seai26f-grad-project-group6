from step18 import fix as _fix


def fix(train_csv, row, frozen=()) -> dict:
    """Return the same row's inputs with as few values changed as possible so the row becomes good.

    `row` is a bad row with its goals removed. "Good" is the verdict step6.why
    uses: the average standardized goal score is at least the training median.
    Columns named in `frozen` are never changed. A single column name may be
    passed as a plain string. Covers both client questions: the smallest nudge
    that brings a drifting customer back, and the offer that would turn a
    deal-only shopper into a regular one (pass columns that must stay as they
    are in `frozen`).
    """
    if isinstance(frozen, str):
        frozen = (frozen,)
    return _fix(train_csv, row, tuple(frozen))

