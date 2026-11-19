import bisect
import os
import random

import pytest

from choose import pick
from domains.players import step11
from table import Table


@pytest.fixture(scope="module")
def players():
    if not os.path.exists(step11.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step11.DATA, step11.SKIP)


def test_pick_with_20_trips_finds_a_top_player_and_beats_random_on_average(players):
    ranked = sorted(map(players.disty, players.rows))
    rank = lambda r: bisect.bisect_left(ranked, players.disty(r)) / len(ranked)
    tuned = [rank(pick(players, random.Random(s), 20)) for s in range(15)]
    chance = [rank(min(random.Random(s).sample(players.rows, 20), key=players.disty)) for s in range(15)]
    assert sorted(tuned)[7] < 0.05 and sum(tuned) < sum(chance)


def test_pick_stays_inside_the_rows_it_is_given_and_inside_the_budget(players):
    age = next(c.at for c in players.cols if c.name == "Age")
    young = [r for r in players.rows if r.cells[age] < 24]
    calls = []
    def score(r):
        calls.append(r)
        return players.disty(r)
    best = pick(players, random.Random(1), 20, score, rows=young)
    assert best.cells[age] < 24 and len(calls) == 20 and all(r.cells[age] < 24 for r in calls)
