# step3.py
"""Step 3: certify whether two sets of measurements really differ.

Uses only the standard library plus ezr.
"""
import ezr


def same(xs, ys) -> bool:
  """True if XS and YS are NOT really different (the gap is noise),
  False if the difference is real.  Uses ezr's cliffs and
  Kolmogorov-Smirnov tests, which must both agree the samples match."""
  xs, ys = list(xs), list(ys)
  if not xs or not ys: return True     # no evidence of a real difference
  return ezr.same(xs, ys)

