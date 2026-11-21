# step3.py
#!/usr/bin/env python3 -B
"Certify whether two sets of measurements differ, via ezr."
import ezr

def same(xs, ys) -> bool:
  """True if XS and YS are NOT really different (the gap is noise),
  False if the difference is real. Uses ezr's cliffs and
  Kolmogorov-Smirnov tests; with no data on either side, there is
  no evidence of a difference, so it returns True."""
  xs, ys = list(xs), list(ys)
  if not xs or not ys: return True
  return ezr.same(xs, ys)

