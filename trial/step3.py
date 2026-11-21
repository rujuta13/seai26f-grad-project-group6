# step3.py
"Step 3: certify whether two sets of run times really differ."
import ezr

def same(xs, ys) -> bool:
  """True if XS and YS are NOT really different (the gap is noise),
  False if the difference is real. Both ezr tests (Cliff's delta and
  Kolmogorov-Smirnov) must find no difference for this to return True."""
  xs = [float(x) for x in xs]
  ys = [float(y) for y in ys]
  if not xs or not ys:
    raise ValueError("same: need at least one number in each sample")
  return bool(ezr.same(xs, ys))

