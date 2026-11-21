"""step19.py: spot. Flag player records that look like data-entry mistakes, and
players so unusual that their ratings should be checked before they are trusted.

The suspicion score is step9.odd: an input part (how far a flag set lies from
the nearest training flag set) plus a goal part (how far each goal value lies
from what training players with the same flags measured). This module reuses
that scorer and only makes sure every score is a finite float.

Reading, column splitting, encoding, the reference limit, robust goal scales
and the scoring itself come from steps 1, 2, 4 and 9.
"""
from __future__ import annotations

import math

from step9 import CAP
from step9 import odd as _odd


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included). Higher means more
    likely a glitch, a typo, or a flag set unlike anything in train_csv.

    Each score is a finite, non-negative float, capped at CAP per part. Scores
    of 0.0 mean nothing unusual was found, or that there were too few training
    rows to compare against.
    """
    out: list[float] = []
    for s in _odd(train_csv, rows):
        v = float(s)
        out.append(v if math.isfinite(v) else float(CAP))
    return out

