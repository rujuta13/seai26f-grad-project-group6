# step3.py: certify whether two sets of numbers really differ.
"step3.py: certify whether two sets of numbers really differ."
import ezr

def same(xs: list, ys: list) -> bool:
  "True if XS and YS are not really different (the gap is noise), False if the difference is real."
  xs = [float(x) for x in xs if x != "?"]
  ys = [float(y) for y in ys if y != "?"]
  if not xs or not ys: return True     # no evidence of any difference
  return ezr.same(xs, ys)              # cliffs and ks: both must call them alike

