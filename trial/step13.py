# step13.py
#!/usr/bin/env python3 -B
"""
step13.py: certify.  Is a shortlist really better than the league average,
or is that luck?  Are strikers really better penalty takers than midfielders,
or does it just look that way?  same() answers both: True means the gap is
noise, False means the difference is real.  It reuses step3.same, which wraps
ezr's Cliff's delta and Kolmogorov-Smirnov tests.  An empty sample has no
evidence of a difference, so it counts as noise.
"""
import step3

def same(xs, ys) -> bool:
  "True if XS and YS are NOT really different (the gap is noise), False if the difference is real."
  xs, ys = list(xs), list(ys)
  if not xs or not ys: return True
  return step3.same(xs, ys)

