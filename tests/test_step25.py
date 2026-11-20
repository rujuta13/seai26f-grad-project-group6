import os
import random

import pytest

from blame import blame
from domains.marketing import step21
from table import Table


@pytest.fixture(scope="module")
def customers():
    if not os.path.exists(step21.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step21.DATA, step21.SKIP)


def groups(t):
    at = {c.name: c.at for c in t.cols}
    camps = ["ACCEPTEDCMP1", "ACCEPTEDCMP2", "ACCEPTEDCMP3", "ACCEPTEDCMP4", "ACCEPTEDCMP5", "RESPONSE+"]
    said = lambda r: sum(r.cells[at[c]] for c in camps)
    return [r for r in t.rows if said(r) == 0], [r for r in t.rows if said(r) >= 2]


def test_never_responders_have_less_income_and_spend_and_more_children(customers):
    never, often = groups(customers)
    found = {n: d for n, d, ok in blame(customers, never, often, random.Random(1)) if ok}
    assert found["INCOME"] < 0 and found["MNTWINES"] < 0 and found["KIDHOME"] > 0


def test_a_group_against_itself_is_rarely_called_real(customers):
    fake = total = 0
    for seed in range(1, 6):
        rows = random.Random(seed).sample(customers.rows, 1000)
        found = blame(customers, rows[:500], rows[500:], random.Random(seed + 10))
        fake, total = fake + sum(ok for _, _, ok in found), total + len(found)
    assert fake / total < 0.05
