# step23.py
"""Certify for the client: did the last campaign really lift purchases, and do
families with teenagers really shop differently, or does it just look that way?

same() is step13.same: True when the gap is noise, False when it is real.
Unknown cells ('?') are dropped first, and numbers arriving as text are
converted. Both client questions are two-sample comparisons, so the same
test answers each.
"""
import step13


def same(xs, ys) -> bool:
  "True if XS and YS are not really different (the gap is noise); False if the difference is real."
  return step13.same(xs, ys)

