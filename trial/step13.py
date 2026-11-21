"""Step 13: certify. Is a difference between two groups real, or just noise?

Standard library only. Reuses the two-sided permutation test from step3, which
compares the difference of means with random relabellings of the pooled
values and needs no normality assumption. Serves both client questions:
  - is the shortlist's goal values better than the league average?
  - do strikers take better penalties than midfielders?
"""

import step3


def same(xs, ys) -> bool:
    """True if the gap between xs and ys is noise, False if it is a real difference."""
    return step3.same(xs, ys)

