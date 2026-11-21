"""Step 29: flag customer records that look wrong, or behave unlike anyone else.

odd(train_csv, rows) gives one suspicion score per row, goals included. Higher
means more likely a data-entry glitch (an impossible age, an odd income, a
typo) or a record unlike anything in train_csv. Three parts are added:

1. Robust outliers. Every column that is numeric in the training file (goals
   and inputs alike) is centred on its training median and scaled by its
   training median absolute deviation (times 1.4826, so it matches a standard
   deviation for normal data). The part of the robust z-score beyond _TOL is
   added, so an age of 300 or an income of 10 million scores high even when
   no other training row is close to it.
2. Typos and gaps. In a numeric column, a non-numeric text cell adds _BAD_CELL
   (a value that did not parse is a likely typo). A blank cell adds _BLANK_CELL,
   a smaller penalty, since blanks can be genuine missing data.
3. Novelty. Inputs are compared with every training row (numeric inputs scaled
   by their training range, other values must match exactly). The distance to
   the nearest training row is divided by the typical leave-one-out
   nearest-neighbour distance among training rows, so about 1 is ordinary and
   larger means the record is unlike anything measured. This also catches
   misspelt categories, which match no training level.

The scores rank records for checking first. They are heuristics, not calibrated
probabilities, and a high score is not evidence of fraud on its own: an unusual
customer and a mistyped one look alike in the data.
"""
import math

from step1 import _load

_TOL = 3.0            # robust z-scores up to this are normal variation
_BAD_CELL = 10.0      # penalty for a non-numeric text cell in a numeric column
_BLANK_CELL = 1.0     # penalty for a blank cell in a numeric column
_NUM_SHARE = 0.9      # a column is numeric if this share of its training cells is numeric
_PROBES = 200         # training rows used to estimate typical nearest-neighbour distance
_MAD_TO_SD = 1.4826   # scales a median absolute deviation to a standard deviation
_MEAN_AD_TO_SD = 1.2533


def _is_goal(c):
    return c.endswith("+") or c.endswith("-")


def _is_input(c):
    return not (_is_goal(c) or c.endswith("X"))


def _cell(v):
    """Classify a raw cell as ('num', float), ('blank', None) or ('bad', None)."""
    if v is None:
        return "blank", None
    if isinstance(v, float):
        return ("num", v) if math.isfinite(v) else ("bad", None)
    if isinstance(v, str):
        return ("blank", None) if not v.strip() else ("bad", None)
    return "bad", None


def _level(v):
    """Comparable key for a cell: finite float, stripped text, or None if blank."""
    kind, x = _cell(v)
    if kind == "num":
        return x
    if kind == "bad" and isinstance(v, str):
        return v.strip()
    return None


def _median(vals):
    """Median of a non-empty list."""
    s = sorted(vals)
    m = len(s)
    mid = m // 2
    if m % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _numeric_columns(header, train):
    """Return {column: finite training values} for columns that are numeric in training."""
    out = {}
    for c in header:
        if c.endswith("X"):
            continue
        cells = [_cell(r.get(c)) for r in train]
        present = sum(1 for k, _ in cells if k != "blank")
        nums = [x for k, x in cells if k == "num"]
        if present and len(nums) >= _NUM_SHARE * present:
            out[c] = nums
    return out


def _col_stats(vals):
    """Return (median, robust scale) for a non-empty list of finite values."""
    med = _median(vals)
    scale = _MAD_TO_SD * _median([abs(v - med) for v in vals])
    if scale <= 0.0:
        # Fall back to mean absolute deviation, scaled to sd for normal data.
        scale = _MEAN_AD_TO_SD * sum(abs(v - med) for v in vals) / len(vals)
    if scale <= 0.0:
        # Constant column: any real change is still a large deviation.
        scale = 0.01 * max(abs(med), 1.0)
    return med, scale


def _robust_part(row, stats):
    """Outlier and typo penalty for one row over the numeric columns."""
    total = 0.0
    for c, (med, scale) in stats.items():
        kind, x = _cell(row.get(c))
        if kind == "bad":
            total += _BAD_CELL
        elif kind == "blank":
            total += _BLANK_CELL
        else:
            z = abs(x - med) / scale
            if z > _TOL:
                total += z - _TOL
    return total


def _dist(a, b, inputs, spans):
    """Mean per-column difference in [0, 1] between two level dicts."""
    if not inputs:
        return 0.0
    d = 0.0
    for c in inputs:
        x = a[c]
        y = b[c]
        if isinstance(x, float) and isinstance(y, float):
            s = spans[c]
            if s > 0.0:
                d += min(1.0, abs(x - y) / s)
            elif x != y:
                d += 1.0
        elif x != y:
            d += 1.0
    return d / len(inputs)


def _typical_nn(tl, inputs, spans):
    """Typical leave-one-out nearest-neighbour distance among training rows.

    Estimated on a spread of probe rows and floored so duplicate flag sets do
    not make the novelty ratio blow up.
    """
    n = len(tl)
    if not inputs or n < 2:
        return 1.0
    step = max(1, n // _PROBES)
    probes = list(range(0, n, step))[:_PROBES]
    nn = []
    for p in probes:
        best = math.inf
        for j in range(n):
            if j != p:
                d = _dist(tl[p], tl[j], inputs, spans)
                if d < best:
                    best = d
        if best < math.inf:
            nn.append(best)
    floor = 0.5 / len(inputs)
    return max(_median(nn) if nn else 0.0, floor)


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included), in input order.

    Higher means more likely a glitch, a typo, or a record unlike anything in
    train_csv. Each score is the sum of a robust outlier and typo part and an
    input novelty part (see the module docstring). An empty `rows` gives [].
    """
    header, train = _load(train_csv)
    rows = list(rows)
    if not rows:
        return []

    inputs = [c for c in header if _is_input(c)]
    numeric = _numeric_columns(header, train)
    stats = {c: _col_stats(vals) for c, vals in numeric.items()}

    spans = {}
    for c in inputs:
        nums = [v for v in (_level(r.get(c)) for r in train) if isinstance(v, float)]
        spans[c] = (max(nums) - min(nums)) if nums else 0.0

    tl = [{c: _level(r.get(c)) for c in inputs} for r in train]
    typical = _typical_nn(tl, inputs, spans) if (inputs and tl) else 1.0

    out = []
    for r in rows:
        total = _robust_part(r, stats)
        if inputs and tl:
            rl = {c: _level(r.get(c)) for c in inputs}
            nn = min(_dist(rl, t, inputs, spans) for t in tl)
            total += nn / typical
        out.append(float(total))
    return out

