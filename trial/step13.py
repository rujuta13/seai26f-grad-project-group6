# step13.py
"""Step 13: certify.  Are two groups really different, or is the gap noise?

Uses only the standard library plus ezr.
"""
import ezr


def _numbers(xs) -> list[float]:
  "The numeric values of XS, sorted; unknown or text cells are dropped."
  out = []
  for v in xs:
    if isinstance(v, str):
      try: v = float(v)
      except ValueError: continue          # '?' and other text: unknown
    if isinstance(v, (int, float)) and not isinstance(v, bool):
      out.append(float(v))
  return sorted(out)


def same(xs, ys) -> bool:
  """True if XS and YS are NOT really different (the gap is noise),
  False if the difference is real.  Uses ezr's cliffs and
  Kolmogorov-Smirnov tests, which must both agree the samples match."""
  xs, ys = _numbers(xs), _numbers(ys)
  if not xs or not ys: return True     # no evidence of a real difference
  return ezr.same(xs, ys)

