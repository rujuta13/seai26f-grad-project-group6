# step13.py
import ezr

def same(xs, ys) -> bool:
  "True if XS and YS are NOT really different (the gap is noise); False if the difference is real."
  xs = [float(x) for x in xs]
  ys = [float(y) for y in ys]
  if not xs or not ys:
    raise ValueError("same: both lists need at least one number")
  return ezr.same(xs, ys)

