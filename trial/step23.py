# step23.py
"""Step 23: certify.  Did the last campaign really lift purchases, or is
that noise?  Do families with teenagers really shop differently?

Uses only the standard library plus ezr.
"""
import math
import ezr


def same(xs, ys) -> bool:
  """True if XS and YS are NOT really different (the gap is noise),
  False if the difference is real.  Uses ezr's cliffs and
  Kolmogorov-Smirnov tests, which must both agree the samples match."""
  def numbers(vs) -> list[float]:
    "The finite numeric values of VS, sorted; unknown or text cells are dropped."
    out = []
    for v in vs:
      if isinstance(v, str):
        try: v = float(v)
        except ValueError: continue      # '?' and other text: unknown
      if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = float(v)
        if math.isfinite(v): out.append(v)
    return sorted(out)

  xs, ys = numbers(xs), numbers(ys)
  if not xs or not ys: return True       # no evidence of a real difference
  return ezr.same(xs, ys)

