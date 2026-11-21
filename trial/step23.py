"""step23.py: certify, for the two client questions.

Q1: "Did the last campaign really lift purchases, or is that noise?"
Q2: "Do families with teenagers really shop differently, or does it just look that way?"

Both questions are two-sample comparisons, so each is answered by step3.same:
a practical noise-floor test plus a seeded bootstrap interval for the gap in
means. Values that are not finite numbers are dropped first, as in step3.
This module reuses step3.same and only fixes the return type.
"""
from __future__ import annotations

from step3 import same as _same


def same(xs, ys) -> bool:
    """True if the samples xs and ys are NOT really different (the gap is noise).

    xs, ys: lists of numbers (e.g. purchase counts before and after a campaign,
            or purchase counts for families with and without teenagers).
    Returns False only when the difference in means is both larger than the
    benchmark noise floor and clear of zero in a seeded bootstrap interval.
    With fewer than two usable values in either sample there is no evidence
    of a difference, so the result is True.
    """
    return bool(_same(xs, ys))

