import os
import random
import statistics

import pytest

from blame import blame
from domains.marketing import step21, step26
from justify import one_line
from table import Table


@pytest.fixture(scope="module")
def setup():
    if not os.path.exists(step21.DATA):
        pytest.skip("run `make data` first")
    t = Table.read(step21.DATA, step21.SKIP)
    ranked = sorted(t.rows, key=t.disty)
    k = len(ranked) // 10
    return t, ranked, blame(t, ranked[-k:], ranked[:k], random.Random(1)), statistics.median(map(t.disty, t.rows))


def test_skipped_columns_never_appear_as_reasons(setup):
    t, ranked, found, median = setup
    for r in ranked[-60:]:
        line = one_line(t, r, found, median, k=9, skip=step26.FIXED)
        assert not any(name in line for name in step26.FIXED)
        assert line.startswith("Rated bad")


def test_the_worst_customer_is_given_low_spend_reasons(setup):
    t, ranked, found, median = setup
    line = one_line(t, ranked[-1], found, median, k=5, skip=step26.FIXED)
    for part in line[11:-1].split("), "):
        if "average" in part:
            assert float(part.split(" ")[-3]) < float(part.split("average ")[1].rstrip(")"))
