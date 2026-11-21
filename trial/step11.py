import step1


def families(train_csv) -> list[int]:
    """One family (cluster) label per data row of train_csv, in file order."""
    return [int(c) for c in step1.families(train_csv)]


def best(train_csv) -> list[int]:
    """Row indices of the unusually good rows (best on all goals together),
    at most 5% of rows."""
    return sorted(int(i) for i in step1.best(train_csv))

