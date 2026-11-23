"""Step 28: smallest input changes that bring a drifting customer back to a good outcome."""

from step18 import fix as _fix


def fix(train_csv, row, frozen=()) -> dict:
    """Return the row's inputs with as few values changed as possible so it becomes good.

    `row` is a bad row with its goal values removed; any goal keys present are
    ignored. The result contains only input columns, each either unchanged from
    `row` or taken from an observed training build that is good by the step6
    criterion. Columns named in `frozen` are never changed. Among admissible
    candidates, the one differing from `row` in the fewest inputs is chosen,
    with ties going to the higher average percentile and then to the lower
    training index. If no admissible good build exists, the row's inputs are
    returned unchanged.
    """
    return _fix(train_csv, row, frozen=tuple(frozen))

