# step3.py
"""
step3: certify.  Are two samples of numbers really different, or is the
gap just run-to-run noise?  Reuses ezr's same (cliffs and ks, the
two-sample tests) rather than rewriting them.
"""
import ezr


def same(xs, ys) -> bool:
  """True if XS and YS are NOT really different (the gap is noise),
  False if the difference is real.  Both are lists of numbers."""
  xs = [float(x) for x in xs]
  ys = [float(y) for y in ys]
  if not xs or not ys:
    raise ValueError("same: each sample needs at least one number")
  return bool(ezr.same(xs, ys))

