import os
import random

import pytest

from certify import gap
from domains.marketing import step21, step23
from table import Table


@pytest.fixture(scope="module")
def customers():
    if not os.path.exists(step21.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step21.DATA, step21.SKIP)


def test_responders_buy_more_by_catalogue_and_web_than_non_responders(customers):
    at = {c.name: c.at for c in customers.cols}
    yes = [r for r in customers.rows if r.cells[at["RESPONSE+"]] == 1]
    no = [r for r in customers.rows if r.cells[at["RESPONSE+"]] == 0]
    rng = random.Random(1)
    got = dict(zip([c.name for c in customers.y], gap(customers, rng.sample(yes, 100), rng.sample(no, 100))))
    for name in ("NUMCATALOGPURCHASES+", "NUMWEBPURCHASES+"):
        a, b, noise = got[name]
        assert not noise and a > b
    assert got["COMPLAIN-"][2]  # complaints are too rare to tell apart


def test_two_random_halves_of_the_customers_are_noise(customers):
    rows = random.Random(2).sample(customers.rows, 200)
    assert sum(g[2] for g in gap(customers, rows[:100], rows[100:])) >= 7  # at most one stray in eight goals
