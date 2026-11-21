# step13.py
"""Certify whether two samples really differ.

Used for: is the shortlist really better than the league average, and
are strikers really better penalty takers than midfielders?

same() is True when the gap is noise, False when it is real. Unknown
cells ('?') are not evidence, so they are dropped before testing. Numbers
arriving as text are converted first. The test itself is step3.same,
which asks Cliff's delta and Kolmogorov-Smirnov.
"""
import step2
import step3


def same(xs, ys) -> bool:
  "True if XS and YS are not really different (the gap is noise); False if the difference is real."
  xs = [v for v in map(step2._cell, xs) if step2._known(v)]
  ys = [v for v in map(step2._cell, ys) if step2._known(v)]
  return step3.same(xs, ys)

