import os
import random

import pytest

from domains.marketing import step21
from guess import fit, guess
from table import Table


@pytest.fixture(scope="module")
def customers():
    if not os.path.exists(step21.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step21.DATA, step21.SKIP)


def test_channel_guesses_beat_the_average_on_unseen_customers(customers):
    rng = random.Random(1)
    rows = rng.sample(customers.rows, len(customers.rows))
    train, test = rows[:1764], rows[1764:]
    tree = fit(customers, train, rng)
    out = [guess(customers, tree, r) for r in test]
    names = [c.name for c in customers.y]
    for goal in ("NUMCATALOGPURCHASES+", "NUMSTOREPURCHASES+", "NUMWEBPURCHASES+"):
        i, c = names.index(goal), customers.y[names.index(goal)]
        mean = sum(r.cells[c.at] for r in train) / len(train)
        assert sum(abs(g[i][0] - r.cells[c.at]) for g, r in zip(out, test)) < 0.9 * sum(abs(mean - r.cells[c.at]) for r in test)


def test_the_customers_guessed_most_likely_to_respond_do_respond_more(customers):
    rng = random.Random(1)
    rows = rng.sample(customers.rows, len(customers.rows))
    train, test = rows[:1764], rows[1764:]
    tree = fit(customers, train, rng)
    i = [c.name for c in customers.y].index("RESPONSE+")
    at = customers.y[i].at
    ranked = sorted(test, key=lambda r: -guess(customers, tree, r)[i][0])
    top = ranked[:len(ranked) // 5]
    assert sum(r.cells[at] for r in top) / len(top) > 1.5 * sum(r.cells[at] for r in test) / len(test)
