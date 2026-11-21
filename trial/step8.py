"""step8.py: the smallest change to a bad build that makes it good, leaving frozen settings alone.

The fix is a measured build from the training data, so its goals are known rather than predicted.
"""

from step1 import _load, best
from step5 import _goal_scores
from step6 import _num


def _same_value(a, b) -> bool:
    """Equal as numbers when both parse as numbers (so '1' matches 1.0), otherwise as strings."""
    fa, fb = _num(a), _num(b)
    if fa is not None and fb is not None:
        return fa == fb
    return str(a) == str(b)


def fix(train_csv, row, frozen=()) -> dict:
    """Return the bad row's inputs with as few values changed as possible so the build is good.

    Candidates are measured training builds that agree with `row` on every frozen column,
    so no frozen setting changes. Among the good candidates (the Pareto front, or the ranked
    subset when the front is large), the one that changes the fewest settings wins, with
    the best combined goal score breaking ties. If no good candidate is eligible, the
    eligible build with the best combined score is returned instead.

    Raises ValueError if `frozen` names a column that is not an input, or if no measured
    build has the frozen settings.
    """
    rows, inputs, goals = _load(train_csv)
    frozen = set(frozen)

    unknown = frozen - set(inputs)
    if unknown:
        raise ValueError(f"frozen columns are not inputs: {sorted(unknown)}")

    base = {c: row.get(c, "") for c in inputs}
    if not rows or not goals:
        raise ValueError("train_csv has no measured builds with goal columns")

    score = _goal_scores(rows, goals)
    good = set(best(train_csv))

    eligible = [
        i for i in range(len(rows))
        if all(_same_value(rows[i][c], base[c]) for c in frozen)
    ]
    if not eligible:
        raise ValueError("no measured build has the frozen settings, so no fix is possible")

    def changes(i):
        return sum(0 if _same_value(rows[i][c], base[c]) else 1 for c in inputs)

    good_eligible = [i for i in eligible if i in good]
    if good_eligible:
        # Fewest changed settings first, then the best combined score.
        pick = min(good_eligible, key=lambda i: (changes(i), -score[i], i))
    else:
        pick = max(eligible, key=lambda i: (score[i], -changes(i), -i))

    out = {}
    for c in inputs:
        if _same_value(rows[pick][c], base[c]):
            out[c] = base[c]
        else:
            out[c] = rows[pick][c]
    return out

