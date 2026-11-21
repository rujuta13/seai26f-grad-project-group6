"""Step 19: spot. Flag player records that look like data-entry mistakes.

Serves the client's requests:
  - "Flag player records that look like data-entry mistakes."
  - "Which players are so unusual we should look twice before trusting their ratings?"

Thin wrapper around step9.odd. Each score is the largest of three signals (an
unfamiliar flag set, an odd goal value, or disagreement with exact flag-set
twins) plus a small share of their average, so higher means look twice. A
missing or non-numeric goal cell is treated as suspicious. Any NaN score is
reported as fully suspicious (1.0), so no record goes unflagged by accident.
"""

import math

import step9


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in `rows` (goals included), in order.

    Higher means more likely a glitch, a typo, or a flag set unlike anything in
    train_csv. Scores lie in [0, 1.1]. A NaN score is replaced by 1.0 so that
    an undecidable record is flagged for review.
    """
    scores = step9.odd(train_csv, rows)
    out = []
    for s in scores:
        s = float(s)
        out.append(1.0 if math.isnan(s) else s)
    return out

