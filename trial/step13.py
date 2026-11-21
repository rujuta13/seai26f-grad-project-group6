#!/usr/bin/env python3 -B
"step13.py: certify. Is a gap between two samples real, or just luck?"
from step3 import same as _same

def same(xs: list[float], ys: list[float]) -> bool:
  "True if XS and YS are NOT really different (the gap is noise), False if the difference is real."
  if not xs or not ys: raise ValueError("need at least one number in each sample")
  return _same(list(xs), list(ys))

