import step9


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included).

    Higher means more likely a glitch, a typo, or unlike anything in
    train_csv. Delegates to step9.odd, which penalises missing or non-finite
    cells, unseen categories, out-of-range inputs, goal values far from the
    training bulk, and input configurations far from every training row.
    """
    return [float(s) for s in step9.odd(train_csv, list(rows))]

