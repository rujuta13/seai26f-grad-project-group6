import os
import random
import statistics

import pytest

from blame import blame
from domains.players import step11
from justify import one_line
from table import Table


@pytest.fixture(scope="module")
def setup():
    if not os.path.exists(step11.DATA):
        pytest.skip("run `make data` first")
    t = Table.read(step11.DATA, step11.SKIP)
    ranked = sorted(t.rows, key=t.disty)
    k = len(ranked) // 10
    return t, ranked, blame(t, ranked[-k:], ranked[:k], random.Random(1)), statistics.median(map(t.disty, t.rows))


def test_best_player_is_rated_good_with_high_ratings_as_reasons(setup):
    t, ranked, found, median = setup
    line = one_line(t, ranked[0], found, median, k=5)
    assert line.startswith("Rated good: ") and "average" in line
    at = {c.name: c.at for c in t.x}
    for part in line[12:-1].split("), "):
        name, rest = part.split(" ", 1) if part.split(" ")[0] in at else (None, None)
        if name:
            assert float(rest.split(" ")[0]) > float(rest.split("average ")[1].rstrip(")"))  # above average


def test_worst_player_is_rated_bad_with_low_ratings_as_reasons(setup):
    t, ranked, found, median = setup
    line = one_line(t, ranked[-1], found, median)
    assert line.startswith("Rated bad: ") and "no single column" not in line
    for part in line[11:-1].split("), "):
        value, avg = part.split(" ")[-3], part.split("average ")[1].rstrip(")")
        assert float(value) < float(avg)
