#!/usr/bin/env python3 -B
"step23.py: certify. Did the last campaign really lift purchases, and do families with teenagers really shop differently, or is that noise?"
from step13 import same as _same

def same(xs: list[float], ys: list[float]) -> bool:
  "True if XS and YS are NOT really different (the gap is noise), False if the difference is real."
  return _same(xs, ys)

