# step3.py
import ezr

def same(xs, ys) -> bool:
  "True if XS and YS are not really different (the gap is noise), False if the difference is real."
  xs, ys = list(xs), list(ys)
  if not xs or not ys: raise ValueError("same: both samples need at least one number")
  return bool(ezr.same(xs, ys))

