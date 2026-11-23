from step5 import blame as _blame


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Association in the observed builds, not proof of cause. Goals are read
    from the file's column suffixes: "+" goals are maximised, "-" goals are
    minimised, and "X" columns are ignored.
    """
    return _blame(train_csv)

