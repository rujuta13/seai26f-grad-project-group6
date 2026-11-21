import step15


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Delegates to step15.blame. Bad rows are the worst quarter of the training
    rows, ranked by the sum of their per-goal ranks across all goal columns.
    This signature cannot select a single goal, so the ranking reflects overall
    badness, not campaign response or complaints alone. Inputs that separate
    the bad rows most come first.
    """
    return list(step15.blame(train_csv))

