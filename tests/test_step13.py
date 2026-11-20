import os
import random

import pytest

from certify import gap, say
from domains.players import step11, step13
from guess import fit
from table import Table


def test_say_words():
    assert say((1, 2, True), "a") == "noise"
    assert say((1, 2, False), "a") == "real: a is 50.0% lower"
    assert say((3, 2, False), "a") == "real: a is 50.0% higher"


@pytest.fixture(scope="module")
def players():
    if not os.path.exists(step11.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step11.DATA, step11.SKIP)


def test_a_shortlist_picked_without_seeing_goals_is_really_better(players):
    rng = random.Random(1)
    rows = rng.sample(players.rows, 6000)
    train, test = rows[:4800], rows[4800:]
    tree = fit(players, train, rng)
    short = sorted(test, key=lambda r: step13.predicted(players, tree, r))[:50]
    for (a, b, noise), c in zip(gap(players, short, rng.sample(test, 100)), players.y):
        assert not noise and a > b  # both goals are better when higher


def test_two_random_halves_of_the_players_are_noise(players):
    rows = random.Random(2).sample(players.rows, 200)
    assert all(g[2] for g in gap(players, rows[:100], rows[100:]))


def test_say_handles_a_zero_median_and_tied_medians():
    assert say((1, 0, False), "a") == "real: a median 1 vs 0"
    assert say((0, 0, False), "a") == "real: a differ in spread, medians tie at 0"
