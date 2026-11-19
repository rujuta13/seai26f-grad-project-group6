import os
import random

import pytest

from domains.ssw import step1
from flow import Keeper, Watch, worse
from table import Table


@pytest.fixture(scope="module")
def ssw():
    if not os.path.exists(step1.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step1.DATA)


def test_keeper_keeps_mostly_good_builds_and_few_overall(ssw):
    rng = random.Random(1)
    rows = rng.sample(ssw.rows, 2000)
    keeper = Keeper(ssw, share=0.05)
    cut = sorted(map(ssw.disty, ssw.rows))[int(0.05 * len(ssw.rows))]
    kept = [r for r in rows if keeper.add(r)]
    assert 0 < len(kept) < 0.25 * len(rows)
    assert sum(ssw.disty(r) <= cut for r in kept) / len(kept) > 0.8


def test_keeper_says_nothing_during_warm_up(ssw):
    keeper = Keeper(ssw, warm=50)
    assert not any(keeper.add(r) for r in ssw.rows[:50])


def test_watch_stays_quiet_when_nothing_changes(ssw):
    rng = random.Random(2)
    stream = rng.sample(ssw.rows, 1000)
    watch = Watch(ssw, stream[:100])
    assert sum(watch.add(r) for r in stream[100:]) <= 1


def test_watch_alarms_after_a_slowdown(ssw):
    rng = random.Random(3)
    stream = rng.sample(ssw.rows, 1000)
    watch = Watch(ssw, stream[:100])
    assert not any(watch.add(r) for r in stream[100:500])
    assert any(watch.add(worse(ssw, r, 0.15)) for r in stream[500:])
