# step13.py: certify. Is a difference between two sets of numbers real, or just noise?
"step13.py: certify. Is a difference between two sets of numbers real, or just noise?"
import ezr

def same(xs, ys) -> bool:
  "True if XS and YS are not really different (the gap is noise), False if the difference is real."
  xs = [float(x) for x in xs if x != "?"]
  ys = [float(y) for y in ys if y != "?"]
  if not xs or not ys: return True     # no evidence of any difference
  return ezr.same(xs, ys)              # cliffs and ks: both must call them alike

