import step18


def fix(train_csv, row, frozen=()) -> dict:
    """Inputs of a bad customer `row`, changed in as few values as possible so it
    becomes good, with the columns named in `frozen` left exactly as they are.

    `row` carries only input columns (goals removed). The result has the same
    input columns as `row`. Values come from a training row that no other
    training row dominates, chosen by step18.fix (which uses step8.fix). Names
    in `frozen` that are not inputs are ignored. If no training row agrees
    with the frozen columns, the inputs are returned unchanged.
    """
    return step18.fix(train_csv, row, frozen=frozen)

