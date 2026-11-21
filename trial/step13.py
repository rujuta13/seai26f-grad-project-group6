"""step13.py: certify whether two sets of measurements really differ.

Standard library only (Python 3.13). Reuses step3 (Cliff's delta and the
Efron bootstrap test) rather than repeating it.

Both client questions reduce to this check:
  - shortlist versus league average: pass the shortlist's measurements as xs
    and the league's as ys;
  - strikers versus midfielders on penalties: pass the penalty values of each
    group as xs and ys.

Returns True when the gap is noise, and False only when the difference is
both non-negligible in effect size and statistically significant.
"""
from __future__ import annotations

from step3 import same as _step3_same


def same(xs, ys) -> bool:
    """True if xs and ys are NOT really different (the gap is noise).

    xs, ys: lists of numbers (e.g. run times). Returns False only when the
    difference is both non-negligible in effect size and statistically
    significant. Empty input or identical values count as the same.
    """
    return _step3_same(xs, ys)

