# step13.py
#!/usr/bin/env python3 -B
"Certify whether two groups of measurements really differ, via ezr."
import math
import ezr

def _clean(vs) -> list[float]:
  "The usable numbers in VS as floats; unknown, missing, bool, and non-finite cells are dropped."
  out = []
  for v in vs:
    if isinstance(v, str):
      v = ezr.atom(v.strip()) if v.strip() != "?" else None
    if isinstance(v, bool) or not isinstance(v, (int, float)): continue
    if math.isfinite(v): out.append(float(v))
  return out

def same(xs, ys) -> bool:
  """True if XS and YS are NOT really different (the gap is noise),
  False if the difference is real. Uses ezr's cliffs and
  Kolmogorov-Smirnov tests. With no numbers on either side there is
  no evidence of a difference, so it returns True."""
  xs, ys = _clean(xs), _clean(ys)
  if not xs or not ys: return True
  return ezr.same(xs, ys)

