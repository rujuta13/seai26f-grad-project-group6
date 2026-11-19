import os
import random

import pytest

from blame import blame, split
from domains.players import step11
from table import Table


@pytest.fixture(scope="module")
def players():
    if not os.path.exists(step11.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step11.DATA, step11.SKIP)


def test_split_respects_a_higher_is_better_goal(players):
    pen = players.y[0]
    bad, good = split(players, players.rows, pen)
    assert max(r.cells[pen.at] for r in bad) <= min(r.cells[pen.at] for r in good)


def test_weak_penalty_takers_lack_shooting_skills(players):
    found = blame(players, *split(players, players.rows, players.y[0]), random.Random(1))
    real = {n: d for n, d, ok in found if ok}
    assert real["Finishing"] < 0 and real["Long Shots"] < 0


def test_strong_players_have_more_physical_ratings(players):
    rest, strong = split(players, players.rows, players.y[1])
    found = {n: d for n, d, ok in blame(players, rest, strong, random.Random(1)) if ok}
    assert found["PHY"] < 0 and found["Jumping"] < 0


def test_same_group_against_itself_is_rarely_called_real(players):
    """With 42 columns a stray false positive is expected; it should be rare, under 5% of columns."""
    fake = total = 0
    for seed in range(1, 6):
        rows = random.Random(seed).sample(players.rows, 2000)
        found = blame(players, rows[:1000], rows[1000:], random.Random(seed + 10))
        fake, total = fake + sum(ok for _, _, ok in found), total + len(found)
    assert fake / total < 0.05
