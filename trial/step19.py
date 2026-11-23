from step9 import odd as _odd


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included): higher means more likely a
    glitch, a typo, or unlike anything in train_csv.

    Delegates to step9.odd, which adds together flag novelty, goal anomaly (largest
    absolute z-score against the training goals), and heavily weighted bad cells
    (missing, non-numeric, or never-seen flag values).
    """
    rows = list(rows)
    if not rows:
        return []
    return [float(s) for s in _odd(train_csv, rows)]

