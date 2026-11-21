"""Step 29: spot. Flag customer records that look wrong or unlike anyone else.

Serves the client's questions:
  - "Flag customer records that look wrong: impossible ages, odd incomes, typos."
  - "Which customers behave unlike anyone else? Are they fraud, or just unusual?"

Builds on step9.odd, which scores each record for an unfamiliar flag set, odd
goal values, and disagreement with exact flag-set twins. This module adds checks
on the input columns themselves, which step9 does not examine directly:

  1. Impossible values: in a numeric column whose training values are never
     negative, a negative value is flagged outright (for example an age or an
     income below zero).
  2. Odd numeric values: a robust z-score against the training median and MAD,
     as in step9, so an age of 200 or an income far beyond anything seen scores
     as suspicious. Infinite values count as suspicious.
  3. Typos: a non-blank text cell in a column whose training values are mostly
     numbers (for example "3l" in an income column) is flagged. A text value
     never seen in training, in a column with few distinct text values, is
     flagged as weakly suspicious, since it may be a misspelling or a genuinely
     new category.

Blank cells are not treated as errors: missing data is common, and step9
already treats missing goal values as worst.

Each record's score is its largest signal plus a small share of the average
signal, the same rule as step9, so one strong problem flags the record while
weaker signals still rank it above a clean one. Scores lie in [0, 1.1].
"""

import math
import statistics
from functools import lru_cache

import step1
import step9

_Z_LO = 3.0          # robust z-score below which a numeric value is not odd
_Z_SPAN = 7.0        # z-score range over which suspicion rises from 0 to 1
_TYPO = 1.0          # text cell in a column that is otherwise numeric
_IMPOSSIBLE = 1.0    # negative value where training values are never negative
_UNSEEN = 0.3        # text value never seen in a low-cardinality text column
_NUM_SHARE = 0.9     # share of non-blank cells that must be numbers for a numeric column
_MIN_NUM = 2         # fewest numeric training values needed to judge a column
_MAX_CATS = 40       # text columns with more distinct values are not checked
_MIX = 0.1           # weight of the average signal added to the largest one


def _blank(v):
    """True for an empty, missing or NaN cell."""
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    return isinstance(v, float) and math.isnan(v)


@lru_cache(maxsize=4)
def _input_model(train_csv):
    """What the training data says about each input column.

    Returns (inputs, numeric, texts):
      numeric: {name: (median, robust scale, training minimum)} for columns that
               are mostly numbers.
      texts:   {name: frozenset of training values as strings} for columns that
               are mostly text and have few distinct values.
    Columns with no usable training values are left out.
    """
    header, rows = step1._read(train_csv)
    inputs, _, _ = step1._classify(header)
    numeric = {}
    texts = {}
    for name in inputs:
        present = [v for v in (r.get(name) for r in rows) if not _blank(v)]
        if not present:
            continue
        floats = [v for v in present if isinstance(v, float) and math.isfinite(v)]
        if len(floats) >= _MIN_NUM and len(floats) >= _NUM_SHARE * len(present):
            med = statistics.median(floats)
            scale = 1.4826 * statistics.median([abs(x - med) for x in floats])
            if scale <= 0:
                scale = statistics.pstdev(floats)
            numeric[name] = (med, scale, min(floats))
        else:
            keys = {str(v) for v in present}
            if len(keys) <= _MAX_CATS:
                texts[name] = frozenset(keys)
    return inputs, numeric, texts


def _z_suspicion(v, med, scale):
    """Suspicion in [0, 1] for a numeric value against a centre and robust scale."""
    if scale > 0:
        z = (v - med) / scale
        return min(1.0, max(0.0, (abs(z) - _Z_LO) / _Z_SPAN))
    return 0.0 if v == med else 1.0


def _input_signals(row, inputs, numeric, texts):
    """Suspicion signals in [0, 1] from the input cells of one record."""
    signals = []
    for name in inputs:
        if name in numeric:
            v = row.get(name)
            if _blank(v):
                continue
            if not isinstance(v, float) or not math.isfinite(v):
                signals.append(_TYPO)  # text or infinity where numbers belong
                continue
            med, scale, lo = numeric[name]
            if lo >= 0 and v < 0:
                signals.append(_IMPOSSIBLE)
                continue
            signals.append(_z_suspicion(v, med, scale))
        elif name in texts:
            v = row.get(name)
            if _blank(v):
                continue
            if str(v) not in texts[name]:
                signals.append(_UNSEEN)
    return signals


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included), in order.

    Higher means more likely a glitch, a typo, an impossible value, or a record
    unlike anything in train_csv. Scores lie in [0, 1.1]. A record whose step9
    score cannot be computed is flagged as fully suspicious (1.0).
    """
    base = step9.odd(train_csv, rows)
    inputs, numeric, texts = _input_model(train_csv)

    scores = []
    for row, b in zip(rows, base):
        b = float(b)
        if not math.isfinite(b):
            scores.append(1.0)
            continue
        signals = _input_signals(row, inputs, numeric, texts)
        if not signals:
            scores.append(b)
            continue
        parts = [min(1.0, max(0.0, b))] + signals
        top = max(parts)
        mean = sum(parts) / len(parts)
        scores.append(float(top + _MIX * mean))
    return scores

