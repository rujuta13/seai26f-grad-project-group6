"""Step 12: guess. Estimate each goal of an unseen player and say how sure we are.

Standard library only. The estimator, the calibrated confidence band and the
input encoding are reused from step2 (which uses step1). This module adds only
the checks the client needs:

- Only the input columns of the query are used, so goal cells are never read.
- A query that is missing an input column is rejected with a clear error,
  instead of silently filling that column with 0 as step2.guess does.

The band low..high is the 5th to 95th percentile of leave-one-out residuals
measured on the training data, so it is the range the true value should fall
in for about 90% of players like this one.
"""

from step2 import _model, guess as _guess


def guess(train_csv, row) -> dict[str, tuple[float, float, float]]:
    """Estimate every goal for an unseen player.

    `row` holds the input columns only (goal columns removed). Returns
    {goal: (estimate, low, high)} for every goal column of train_csv.
    Raises ValueError if an input column of train_csv is absent from row.
    """
    model = _model(train_csv)
    inputs = model["inputs"]
    goal_names = [name for name, _ in model["goals"]]

    missing = [c for c in inputs if c not in row]
    if missing:
        raise ValueError("row is missing input columns: " + ", ".join(missing))

    query = {c: row[c] for c in inputs}
    result = _guess(train_csv, query)
    return {name: result[name] for name in goal_names}

