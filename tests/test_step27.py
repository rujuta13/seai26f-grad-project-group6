import os
import random

import pytest

from choose import only, pick
from domains.marketing import step21
from table import Table


@pytest.fixture(scope="module")
def customers():
    if not os.path.exists(step21.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step21.DATA, step21.SKIP)


def test_pick_with_50_mailings_finds_a_top_customer_within_budget(customers):
    calls = []
    def score(r):
        calls.append(r)
        return customers.disty(r)
    best = pick(customers, random.Random(1), 50, score)
    ranked = sorted(map(customers.disty, customers.rows))
    assert len(calls) == 50 and ranked.index(customers.disty(best)) / len(ranked) < 0.10


def test_aiming_at_store_or_web_changes_who_is_chosen(customers):
    names = [c.name for c in customers.y]
    store, web = names.index("NUMSTOREPURCHASES+"), names.index("NUMWEBPURCHASES+")
    s = pick(customers, random.Random(1), 50, only(customers, store))
    w = pick(customers, random.Random(1), 50, only(customers, web))
    assert s.cells[customers.y[store].at] > w.cells[customers.y[store].at]
    assert w.cells[customers.y[web].at] > s.cells[customers.y[web].at]
