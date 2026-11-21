"""step15.py: blame the input columns for bad rows.

Standard library only (Python 3.13). Reuses step5, which already implements
this: a row's badness is 1 minus its mean normalised goal score (step1
scaling, 1 is best), and input columns are ranked by how much of the
variation in badness they explain (adjusted correlation ratio).

The client's two questions both concern single goals (Penalties+ for
weakness from the spot, Strength+ for physical strength), but blame()
takes no goal argument, so it ranks by badness over all goals, as step5
does. Read the ranking as overall badness, not as a penalty-only or
strength-only answer.
"""
from __future__ import annotations

from step5 import blame as _step5_blame


def blame(train_csv) -> list[str]:
    """Input column names, most to blame for bad rows (bad on the goals) first.

    A column ranks higher the more its values separate bad rows (high
    badness) from good ones. Columns with no explanatory power keep their
    header order at the end.
    """
    return _step5_blame(train_csv)

