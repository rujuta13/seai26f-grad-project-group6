# step23.py
#!/usr/bin/env python3 -B
"""
step23.py: certify.  Did the last campaign really lift purchases, or is
that noise?  Do families with teenagers really shop differently, or does
it just look that way?  same() answers both: True means the gap is noise,
False means the difference is real.  Nothing new is needed, so this file
delegates to step13.same, which wraps step3 and ezr's Cliff's delta and
Kolmogorov-Smirnov tests, and treats an empty sample as noise.
"""
import step13

def same(xs: list[float], ys: list[float]) -> bool:
  "True if XS and YS are NOT really different (the gap is noise), False if the difference is real."
  return step13.same(xs, ys)

