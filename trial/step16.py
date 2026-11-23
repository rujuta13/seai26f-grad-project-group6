"""Step 16: client-facing justification.

explain(train_csv): plain-language reasons for what makes rows good, each
checkable by a skeptic against the data.

why(train_csv, row): one or two sentences on why this row (goals included)
is rated good or bad.
"""

import csv
import math

import step6


def _goal_columns(train_csv) -> list[str]:
    with open(train_csv, newline="") as f:
        cols = list(csv.DictReader(f).fieldnames or [])
    return [c for c in cols if c.endswith(("+", "-"))]


def _is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def explain(train_csv) -> str:
    """Reasons, in plain words a skeptical expert can check, for what makes
    rows good."""
    return str(step6.explain(train_csv))


def why(train_csv, row) -> str:
    """One or two sentences on why this row (goals included) is good or bad."""
    goals = _goal_columns(train_csv)
    missing = [g for g in goals if not _is_finite_number(row.get(g))]
    if missing:
        return ("Cannot rate this build: goal values are missing for "
                + ", ".join(missing)
                + ", so its quality cannot be judged from these goals.")
    return str(step6.why(train_csv, row))

