import os
import random

import pytest

from cluster import families
from domains.marketing import step21
from remember import spread, traits
from table import Table


@pytest.fixture(scope="module")
def customers():
    if not os.path.exists(step21.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step21.DATA, step21.SKIP)


def test_inputs_exclude_goals_derived_totals_and_constant_columns(customers):
    names = {c.name for c in customers.x}
    assert not names & {"MNTTOTAL", "MNTREGULARPRODS", "ACCEPTEDCMPOVERALL", "Z_COSTCONTACT", "Z_REVENUE"}
    assert not any(n.endswith(("+", "-")) for n in names)
    assert len(customers.x) == 26 and len(customers.y) == 8 and len(customers.rows) == 2205


def test_customer_families_cover_everyone_and_read_clearly(customers):
    gs = families(customers, random.Random(1))
    spec = spread(customers)
    assert 4 <= len(gs) <= 16 and sum(map(len, gs)) == len(customers.rows)
    assert len({traits(customers, g, spec) for g in gs}) == len(gs)


def test_the_best_customers_beat_the_average_on_purchases(customers):
    best = sorted(customers.rows, key=customers.disty)[:22]
    web = customers.y[1].at
    assert sum(r.cells[web] for r in best) / 22 > 1.5 * sum(r.cells[web] for r in customers.rows) / len(customers.rows)
