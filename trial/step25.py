"""Step 25: rank input columns by how strongly they are associated with bad rows.

Serves the client questions about customers who never respond to campaigns and
customers who complain. Goals come from the column suffixes: "+" goals are
maximised, "-" goals are minimised, and "X" columns are ignored. Since
COMPLAIN- and RESPONSE+ are goals, they shape the badness ranking but are not
inputs. Results show association in the observed customers, not proof of cause.
"""

from step5 import blame as _blame


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    Association in the observed builds, not proof of cause. Goals are read
    from the file's column suffixes: "+" goals are maximised, "-" goals are
    minimised, and "X" columns are ignored.
    """
    return list(_blame(train_csv))

