# step3.py
#!/usr/bin/env python3 -B
"""
step3.py: certify.  Are two samples of run times really different, or is
the gap just noise?  Reuses ezr's non-parametric tests: Cliff's delta and
Kolmogorov-Smirnov, which must both agree that the samples are the same.
"""
import ezr

def same(xs: list[float], ys: list[float]) -> bool:
  "True if XS and YS are NOT really different (the gap is noise)."
  return ezr.same(list(xs), list(ys))

