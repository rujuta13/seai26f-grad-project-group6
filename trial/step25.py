"""Step 25: which input columns go with customers who ignore campaigns or complain.

blame(train_csv) ranks input columns for the two client questions:

  - Why some customers never respond to campaigns: the RESPONSE+ goal.
  - What customers who complain have in common: the COMPLAIN- goal.

Each goal is scored on its own, as in step15.blame, and a column's blame is the
higher of its two scores. Badness on a goal is 1 minus its normalised value,
so a column that goes with no response or with complaints scores high. If
neither goal is in the file, the overall goal badness from step5.blame is used
instead. Columns are returned most to blame first, with ties kept in file order.
"""
from step1 import _load
from step5 import _blame_score, _goals, _is_goal, _keys, _scores

_FOCUS = ("RESPONSE", "COMPLAIN")


def _is_input(c):
    return not _is_goal(c) and not c.endswith("X")


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    A column's blame is the largest of its scores over each focus goal
    ('RESPONSE' and 'COMPLAIN') alone. If neither focus goal is present, the
    score is taken over all goals together. Input columns are every column
    that is not a goal ('+' or '-') and not ignored ('X'). Ties keep their
    order in the file.
    """
    header, rows = _load(train_csv)
    inputs = [c for c in header if _is_input(c)]
    if not inputs:
        return []

    all_goals = _goals(header)
    focus = [(c, sg) for c, sg in all_goals if c[:-1] in _FOCUS]
    if focus:
        views = [[g] for g in focus]
    else:
        views = [all_goals] if all_goals else []

    position = {c: p for p, c in enumerate(header)}
    score = {c: 0.0 for c in inputs}
    for goals in views:
        idx, bad = _scores(rows, goals)
        if not idx:
            continue
        for c in inputs:
            raw = [rows[i].get(c) for i in idx]
            s = _blame_score(bad, _keys(raw))
            if s > score[c]:
                score[c] = s

    return sorted(inputs, key=lambda c: (-score[c], position[c]))

