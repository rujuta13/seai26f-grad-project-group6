"""step23.py: certify whether two sets of measurements really differ.

Standard library only (Python 3.13). Reuses step13.same, which reuses
step3 (Cliff's delta and the Efron bootstrap test).

Both client questions reduce to this check:
  - did the last campaign lift purchases: pass purchase counts of customers
    contacted in the campaign as xs and those not contacted as ys;
  - do families with teenagers shop differently: pass the purchase values of
    families with teenagers as xs and those without as ys.

Returns True when the gap is noise, and False only when the difference is
both non-negligible in effect size and statistically significant.
"""
from __future__ import annotations

from step13 import same as _step13_same


def same(xs, ys) -> bool:
    """True if xs and ys are NOT really different (the gap is noise).

    xs, ys: lists of numbers (e.g. run times). Returns False only when the
    difference is both non-negligible in effect size and statistically
    significant. Empty input or identical values count as the same.
    """
    return _step13_same(xs, ys)

