"""step14.py: flow tools for scouting reports arriving one at a time.

stream() keeps a player when no earlier player (training or already streamed) is at
least as good on every goal and strictly better on one. That is step4.stream, reused
unchanged.

drift() warns when the last _WINDOW new players sit significantly further from the
training players than training players sit from each other. It is step4.drift with
the identifier columns (num, Name, url) removed from the inputs, as step11 and step12
do, because a new season's players have new names and would look novel for that
reason alone. The distance and significance test are the same as step4's.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from step1 import _classify, _load
from step3 import same
from step4 import _encode_row, _encode_train, _loo_nearest
from step4 import stream as _stream
from step11 import _trait_inputs

_WINDOW = 30   # number of recent new players tested together for drift


def stream(train_csv, rows) -> list[bool]:
    """Keep (True) or discard (False) each row, in arrival order.

    A row is kept if no training row and no earlier streamed row is at least as
    good on every goal and strictly better on at least one.
    """
    return _stream(train_csv, rows)


def drift(train_csv, rows) -> int | None:
    """Index of the first new row at which the new players stop resembling train_csv.

    Identifier columns are ignored when measuring resemblance. Returns None if no
    warning is raised.
    """
    header, train_rows = _load(train_csv)
    _, _, inputs = _classify(header)
    inputs = _trait_inputs(header, inputs)
    n = len(train_rows)
    if n < 2 or not rows or not inputs:
        return None

    X, specs = _encode_train(train_rows, inputs)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    ref = _loo_nearest(Z)

    new_dist: list[float] = []
    for i, row in enumerate(rows):
        q = _encode_row(row, header, inputs, specs)
        zq = scaler.transform(q.reshape(1, -1))[0]
        d = float(np.sqrt(((Z - zq) ** 2).sum(axis=1)).min())
        new_dist.append(d)
        if len(new_dist) >= _WINDOW and not same(ref, new_dist[-_WINDOW:]):
            return i
    return None

