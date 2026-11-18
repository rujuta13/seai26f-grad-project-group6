import os
import random

import pytest

from blame import blame, split
from domains.ssw import step1
from table import Table


@pytest.fixture(scope="module")
def ssw():
    if not os.path.exists(step1.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step1.DATA)


def test_split_takes_the_tails(ssw):
    bad, good = split(ssw, ssw.rows, ssw.y[0], 0.1)
    assert len(bad) == len(good) == len(ssw.rows) // 10
    assert min(float(r.cells[ssw.y[0].at]) for r in bad) >= max(float(r.cells[ssw.y[0].at]) for r in good)


def test_slow_builds_blame_the_flag_step3_found(ssw):
    found = blame(ssw, *split(ssw, ssw.rows, ssw.y[0]), random.Random(1))
    real = [name for name, _, ok in found if ok]
    assert "inline" in real and "called_value_propagation" in real
    assert found[0][1] > 0 and all(abs(a[1]) >= abs(b[1]) for a, b in zip(found, found[1:]))


def test_blaming_a_group_against_itself_finds_nothing_real(ssw):
    rows = random.Random(2).sample(ssw.rows, 2000)
    assert not any(ok for _, _, ok in blame(ssw, rows[:1000], rows[1000:], random.Random(3)))
