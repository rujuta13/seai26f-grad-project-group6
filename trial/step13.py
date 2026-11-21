# step13.py
"""
step13: certify, for the client.  Is our shortlist really better than the
league average, or is that luck?  Are strikers better penalty takers than
midfielders, or does it just look that way?  Both are two-sample questions,
so this reuses step3.same (ezr's cliffs and ks) rather than rewriting them.
"""
import step3


def same(xs, ys) -> bool:
  """True if XS and YS are NOT really different (the gap is noise),
  False if the difference is real.  Both are lists of numbers, e.g. run times."""
  return step3.same(xs, ys)

