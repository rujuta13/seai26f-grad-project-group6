import os
import random

import pytest

from domains.players import step11
from flow import Keeper, Watch, worse
from table import Table


@pytest.fixture(scope="module")
def players():
    if not os.path.exists(step11.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step11.DATA, step11.SKIP)


def test_worse_pushes_a_higher_is_better_goal_down_and_keeps_whole_numbers_whole(players):
    at = players.y[0].at
    rows = players.rows[:2000]
    made = [worse(players, r, 0.1, random.Random(i)) for i, r in enumerate(rows)]
    assert all(m.cells[at] == int(m.cells[at]) and m.x == r.x for m, r in zip(made, rows))
    before, after = (sum(r.cells[at] for r in g) / len(g) for g in (rows, made))
    assert abs(after - 0.9 * before) < 0.3  # the average moves by 10% although each value stays whole


def test_keeper_keeps_mostly_top_players(players):
    rng = random.Random(1)
    rows = rng.sample(players.rows, 3000)
    keeper = Keeper(players)
    cut = sorted(map(players.disty, players.rows))[int(.05 * len(players.rows))]
    kept = [r for r in rows if keeper.add(r)]
    assert 0 < len(kept) < 0.2 * len(rows)
    assert sum(players.disty(r) <= cut for r in kept) / len(kept) > 0.8


def test_watch_alarms_when_the_new_intake_is_much_weaker(players):
    rng = random.Random(3)
    stream = rng.sample(players.rows, 1500)
    watch = Watch(players, stream[:100])
    assert any(watch.add(worse(players, r, 0.4)) for r in stream[100:])
