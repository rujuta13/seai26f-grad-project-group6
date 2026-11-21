# step23.py
"""
step23: certify, for the client.  Two asks: did the last campaign really
lift purchases, or is that noise?  Do families with teenagers really shop
differently, or does it just look that way?  Both are two-sample questions,
so this reuses step3.same (ezr's cliffs and ks) rather than rewriting them.
"""
import step3


def same(xs, ys) -> bool:
  """True if XS and YS are NOT really different (the gap is noise),
  False if the difference is real.  Both are lists of numbers, e.g. purchase
  counts before and after a campaign, or for families with and without teens."""
  return step3.same(xs, ys)

