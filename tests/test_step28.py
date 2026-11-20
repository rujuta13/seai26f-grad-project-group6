import os
import random

import pytest

from blame import blame, split
from domains.marketing import step21, step28
from fix import improve
from table import Table


@pytest.fixture(scope="module")
def setup():
    if not os.path.exists(step21.DATA):
        pytest.skip("run `make data` first")
    t = Table.read(step21.DATA, step21.SKIP)
    store = [c.name for c in t.y].index("NUMSTOREPURCHASES+")
    found = blame(t, *split(t, t.rows, t.y[store]), random.Random(1))
    good = sorted(t.rows, key=lambda r: -r.cells[t.y[store].at])[:500]
    return t, found, good


def test_offer_plan_only_touches_product_spend(setup):
    t, found, good = setup
    names = [c.name for c in t.x]
    frozen = [n for n in names if not n.startswith("MNT")]
    deal = [r for r in t.rows if r.cells[[c.name for c in t.cols].index("NUMDEALSPURCHASES+")] >= 3][:40]
    plans = [improve(t, r, found, good, frozen) for r in deal]
    plans = [p for p in plans if p]
    assert plans and all(n.startswith("MNT") for p in plans for n, _, _ in p[0])
    assert all(1 <= len(p[0]) <= 3 for p in plans)


def test_the_demo_runs_and_reports_both_asks(setup, capsys):
    step28.main(step21.DATA, 1, 10)
    out = capsys.readouterr().out
    assert "ask 15:" in out and "ask 16:" in out and "reach the target" in out
