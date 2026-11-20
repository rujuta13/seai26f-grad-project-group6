"""The shared split: same rows every time, train and test disjoint, rows copied unmodified."""
import os
from collections import Counter

import pytest

from harness import data


def lines_of(path):
    with open(path, newline="") as f:
        return f.readlines()


@pytest.mark.parametrize("domain", sorted(data.DOMAINS))
def test_split(domain, tmp_path):
    src = os.path.join(data.DATA, data.DOMAINS[domain])
    if not os.path.exists(src):
        pytest.skip("run `make data` first")
    original = lines_of(src)

    (train, n_train), (test, n_test) = data.build(domain, out_dir=tmp_path / "a")
    (train2, _), (test2, _) = data.build(domain, out_dir=tmp_path / "b")

    assert lines_of(train) == lines_of(train2) and lines_of(test) == lines_of(test2)
    assert (n_train, n_test) == (800, 200)
    tr, te = lines_of(train), lines_of(test)
    assert tr[0] == te[0] == original[0]
    # every sampled row is an unmodified source row, and no source row is used twice
    used, have = Counter(tr[1:] + te[1:]), Counter(original[1:])
    assert all(have[r] >= k for r, k in used.items())
