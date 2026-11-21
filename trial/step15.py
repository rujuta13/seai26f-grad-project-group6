"""Step 15: which input columns go with weak penalties and weak physical strength.

blame(train_csv) ranks input columns by how strongly they are linked to bad
rows. Each column gets the highest of several scores:

  - overall: badness on all goals together, as in step5.blame;
  - each focus goal alone: Penalties and Strength, the two the client asked about.

A column that matters for either question therefore ranks high. Badness on a
goal is 1 minus its normalised value, so a column that goes with weak
penalties or weak strength scores high. Its inverse, the columns that go with
strong players, is the same association read from the other side. Columns are
returned most to blame first, with ties kept in file order.
"""
from step1 import _load
from step5 import _blame_score, _goals, _is_goal, _keys, _scores

_FOCUS = ("Penalties", "Strength")


def _is_input(c):
    return not _is_goal(c) and not c.endswith("X")


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows first.

    A column's blame is the largest of its scores over the overall goal
    badness and over each focus goal ('Penalties' and 'Strength') alone.
    Input columns are every column that is not a goal ('+' or '-') and not
    ignored ('X'). Ties keep their order in the file.
    """
    header, rows = _load(train_csv)
    inputs = [c for c in header if _is_input(c)]
    if not inputs:
        return []

    all_goals = _goals(header)
    views = []
    if all_goals:
        views.append(all_goals)
    for c, sg in all_goals:
        if c[:-1] in _FOCUS:
            views.append([(c, sg)])

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

