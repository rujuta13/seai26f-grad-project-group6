"""step9.py: odd. Flag benchmark results that look like glitches, typos, or
flag sets unlike anything seen before.

Each row gets a suspicion score made of two parts that are added together:
  1. Input part: how far its flag set lies from the nearest training flag set,
     as a multiple of the typical nearest-neighbour distance (the same limit
     step4.drift uses). A flag set with no close training neighbour is unusual.
  2. Goal part: how far each goal value lies from what training builds with
     similar flags measured, in robust z-score units. Comparing against
     training builds with the exact same flags catches glitches, since the
     same flags should give about the same result. A missing or non-numeric
     goal counts as a glitch.

Reading, column splitting, number parsing, missing-value filling, input
encoding, the reference limit, the noise floor and quantiles come from steps
1, 2 and 4. Only the scoring is new here.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

from step1 import _SEED, _columns, _dist2, _num, _read
from step2 import NOISE_FLOOR, _fill, _quantile
from step4 import DRIFT_QUANTILE, REFERENCE_ROWS, _encode, _fit

Z_THRESH = 3.5          # robust z-score that counts as one unit of suspicion
CAP = 10.0              # each part of the score is capped so one cell cannot dominate
MIN_GOAL_VALUES = 4     # fewer usable training values for a goal: it cannot be judged
IQR_TO_SIGMA = 1.349    # IQR of a normal distribution is about 1.349 sigma


def _key(vals: list) -> tuple:
    """Hashable key for a row's input cells. Missing or non-finite cells share one key."""
    out = []
    for v in vals:
        if isinstance(v, float) and math.isfinite(v):
            out.append(("f", v))
        elif isinstance(v, str):
            out.append(("s", v))
        else:
            out.append(("n", None))
    return tuple(out)


def odd(train_csv, rows) -> list[float]:
    """One suspicion score per row in rows (goals included). Higher means more
    likely a glitch, a typo, or a flag set unlike anything in train_csv.

    The score is finite and non-negative. Each part is capped at CAP.
    With fewer than two training rows there is nothing to compare against, so
    every score is 0.0.
    """
    header, train = _read(str(train_csv))
    inputs, gmax, gmin = _columns(header)
    goals = gmax + gmin
    if not rows:
        return []
    if len(train) < 2:
        return [0.0 for _ in rows]

    # Input part: encode the training inputs and set a typical neighbour distance.
    spec = _fit(train, inputs)
    enc_train = [_encode(r, inputs, spec) for r in train]
    if len(train) <= REFERENCE_ROWS:
        ref_idx = list(range(len(train)))
    else:
        ref_idx = random.Random(_SEED).sample(range(len(train)), REFERENCE_ROWS)
    ref = [enc_train[i] for i in ref_idx]
    loo = [
        min(_dist2(x, y) for j, y in enumerate(ref) if j != i)
        for i, x in enumerate(ref)
    ]
    limit = max(_quantile(loo, DRIFT_QUANTILE), 1e-9)

    # Training rows grouped by their exact input flags, for the duplicate check.
    dups: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(train):
        dups[_key([r.get(c) for c in inputs])].append(i)

    # Robust centre and scale per goal, from usable training values only.
    gstats: dict[str, tuple[float, float]] = {}
    for g in goals:
        vals = [x for x in (_num(r.get(g)) for r in train) if x is not None]
        if len(vals) < MIN_GOAL_VALUES:
            continue
        med = _quantile(vals, 0.5)
        iqr = _quantile(vals, 0.75) - _quantile(vals, 0.25)
        if iqr > 0:
            scale = iqr / IQR_TO_SIGMA
        else:
            scale = sum(abs(x - med) for x in vals) / len(vals)
        gstats[g] = (med, scale)

    scores: list[float] = []
    for row in rows:
        filled = _fill(row, train, inputs)
        x = _encode(filled, inputs, spec)
        d = min(_dist2(x, y) for y in enc_train)
        input_part = min(CAP, d / limit)

        dup_idx = dups.get(_key([filled.get(c) for c in inputs]), [])
        goal_part = 0.0
        for g, (med, scale) in gstats.items():
            v = _num(row.get(g))
            if v is None:
                # A missing or non-numeric measurement is itself a glitch.
                goal_part = max(goal_part, CAP)
                continue
            sd = max(scale, NOISE_FLOOR * abs(med), 1e-12)
            z = abs(v - med) / sd
            dvals = [
                dv for dv in (_num(train[i].get(g)) for i in dup_idx)
                if dv is not None
            ]
            if dvals:
                # Same flags measured before: a far-off value is suspicious.
                dmed = _quantile(dvals, 0.5)
                z = max(z, abs(v - dmed) / sd)
            goal_part = max(goal_part, min(CAP, z / Z_THRESH))

        scores.append(float(input_part + goal_part))
    return scores

