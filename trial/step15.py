"""Step 15: client-facing blame for builds that score badly on the goals.

blame(train_csv): input column names, most to blame for bad rows first.

Note for the client: blame() ranks inputs against the badness of each build
across all goals together. The two questions ("weak from the spot" and
"physically strong") each concern a single goal, Penalties+ and Strength+.
This signature has no goal-selection parameter, so it cannot answer those
narrower questions directly. A goal-subset version would need its own function
with a goal list parameter, as noted in step11.
"""

from step5 import blame as _blame


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first."""
    return [str(c) for c in _blame(train_csv)]

