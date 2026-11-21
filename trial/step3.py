#!/usr/bin/env python3 -B
"step3.py: are two samples of numbers really different, or just noise?"
from ezr import same as _same

def same(xs: list[float], ys: list[float]) -> bool:
  "True if XS and YS are NOT really different (the gap is noise)."
  if not xs or not ys: raise ValueError("need at least one number in each sample")
  return _same(list(xs), list(ys))

