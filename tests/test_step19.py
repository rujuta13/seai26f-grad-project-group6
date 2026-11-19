import os
import random

import pytest

from cluster import families
from domains.players import step11, step19
from guess import fit
from spot import changed, cutoff, glitch, odd
from table import Table


@pytest.fixture(scope="module")
def players():
    if not os.path.exists(step11.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step11.DATA, step11.SKIP)


def test_swapped_digits_make_a_different_rating_and_leave_single_digits_alone(players):
    r = players.rows[0]
    pen = players.y[0].at
    new = step19.swapped(players, changed(players, r, 71.0))
    assert new.cells[pen] == 17.0
    assert step19.swapped(players, changed(players, r, 55.0)) is None and step19.swapped(players, changed(players, r, 7.0)) is None


def test_typing_mistakes_are_flagged_and_clean_ratings_mostly_are_not(players):
    rng = random.Random(1)
    rows = rng.sample(players.rows, 6000)
    tree = fit(players, rows[:4800], rng)
    test = [r for r in rows[4800:] if 10 < r.cells[players.y[0].at] < 90][:300]
    assert sum(bool(glitch(players, tree, r)) for r in test) / len(test) < 0.05
    assert sum(bool(glitch(players, tree, changed(players, r, 0.0))) for r in test) / len(test) > 0.6
    assert sum(bool(glitch(players, tree, changed(players, r, 99.0))) for r in test) / len(test) > 0.6


def test_unusual_players_are_rarer_in_the_same_pool_than_in_a_different_type_of_player(players):
    gs = sorted(families(players, random.Random(1)), key=lambda g: sum(map(players.disty, g)) / len(g))
    rng = random.Random(2)
    seen = rng.sample(gs[0], 200)
    cut = cutoff(players, seen)
    same = sum(odd(players, seen, r, cut) for r in rng.sample(gs[0], 100))
    other = sum(odd(players, seen, r, cut) for r in rng.sample(gs[-1], 100))
    assert same < 20 and other > same
