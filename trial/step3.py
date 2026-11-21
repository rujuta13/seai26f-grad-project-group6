# step3.py
"""Certify whether two samples (e.g. run times) really differ.

same() is True when the gap is noise, False when it is real.
It reuses ezr.same, which asks Cliff's delta and Kolmogorov-Smirnov
whether the two samples are distinguishable.
"""
import ezr


def same(xs, ys) -> bool:
  "True if XS and YS are not really different (the gap is noise)."
  xs, ys = list(xs), list(ys)
  if not xs or not ys:
    raise ValueError("same: both samples need at least one number")
  return ezr.same(xs, ys)

