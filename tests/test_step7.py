import bisect
import os
import random

import pytest

from choose import only, pick
from domains.ssw import step1
from table import Table


@pytest.fixture(scope="module")
def ssw():
    if not os.path.exists(step1.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step1.DATA)


def test_pick_stays_within_budget_and_finds_a_top_build(ssw):
    calls = []
    def score(r):
        calls.append(r)
        return ssw.disty(r)
    best = pick(ssw, random.Random(1), 30, score)
    assert len(calls) == 30
    ranked = sorted(map(ssw.disty, ssw.rows))
    assert bisect.bisect_left(ranked, ssw.disty(best)) / len(ranked) < 0.10


def test_the_goal_changes_the_choice(ssw):
    speed = pick(ssw, random.Random(1), 30, only(ssw, 0))
    cpu = pick(ssw, random.Random(1), 30, only(ssw, 1))
    assert speed.cells[ssw.y[0].at] < cpu.cells[ssw.y[0].at]
    assert cpu.cells[ssw.y[1].at] < speed.cells[ssw.y[1].at]
