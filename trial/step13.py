"""step13.py: certify. Decide whether two sets of measurements really differ.

Both client questions are two-sample comparisons:
  - Is the shortlist really better than the league average, or is that luck?
  - Are strikers really better penalty takers than midfielders?

same(xs, ys) is True when the gap between the samples is within benchmark
noise, and False when the difference is real. The check itself is step3.same
(a practical noise-floor test plus a seeded bootstrap interval for the gap in
means), so this module reuses it rather than repeating it. Values that are not
finite numbers are dropped first, as in step3.
"""
from __future__ import annotations

from step3 import same as _same


def same(xs, ys) -> bool:
    """True if the samples xs and ys are NOT really different (the gap is noise).

    xs, ys: lists of numbers (e.g. run times, or goal values for two groups).
    Returns False only when the difference in means is both larger than the
    benchmark noise floor and clear of zero in a seeded bootstrap interval.
    With fewer than two usable values in either sample there is no evidence
    of a difference, so the result is True.
    """
    return bool(_same(xs, ys))

