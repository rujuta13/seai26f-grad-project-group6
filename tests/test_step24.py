import os
import random

import pytest

from domains.marketing import step21
from flow import Keeper, Watch, worse
from table import Table


@pytest.fixture(scope="module")
def customers():
    if not os.path.exists(step21.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step21.DATA, step21.SKIP)


def test_keeper_keeps_few_and_mostly_good_customers(customers):
    rng = random.Random(1)
    rows = rng.sample(customers.rows, len(customers.rows))
    keeper = Keeper(customers)
    cut = sorted(map(customers.disty, customers.rows))[int(.05 * len(rows))]
    kept = [r for r in rows if keeper.add(r)]
    assert 0 < len(kept) < 0.15 * len(rows)
    assert sum(customers.disty(r) <= cut for r in kept) / len(kept) > 0.8


def test_watch_alarms_when_deal_purchases_fall_a_lot_and_not_before(customers):
    rng = random.Random(3)
    stream = rng.sample(customers.rows, 1200)
    watch = Watch(customers, stream[:100])
    assert not any(watch.add(r) for r in stream[100:300])
    assert any(watch.add(worse(customers, r, 0.5, rng)) for r in stream[300:])


def test_drifted_counts_stay_whole_numbers(customers):
    at = customers.y[0].at
    assert all(worse(customers, r, 0.15, random.Random(1)).cells[at] == int(worse(customers, r, 0.15, random.Random(1)).cells[at])
               for r in customers.rows[:200])
