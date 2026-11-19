import os
import random

import pytest

from domains.players import step11
from race import bits, forest_guess, verdict
from table import Table


@pytest.fixture(scope="module")
def players():
    if not os.path.exists(step11.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step11.DATA, step11.SKIP)


def test_bits_carry_the_ratings_not_zeros(players):
    row = bits(players, players.rows[:1])[0]
    assert len(row) == len(players.x) and max(row) > 50 and len(set(row)) > 10  # it was all zeros once


def test_forest_learns_penalties_from_the_other_ratings(players):
    rows = random.Random(1).sample(players.rows, 2500)
    train, test = rows[:1000], rows[1000:]
    pen = players.y[0].at
    guesses = forest_guess(players, train, test)
    mean = sum(r.cells[pen] for r in train) / len(train)
    assert sum(abs(g - r.cells[pen]) for g, r in zip(guesses, test)) < 0.6 * sum(abs(mean - r.cells[pen]) for r in test)


def test_verdict_words():
    assert verdict([1, 1, 1, 1], [1, 1, 1, 1], ("a", "b")) == "a vs b: no real difference"
    assert "real, a is better" in verdict([float(i) for i in range(1, 60)], [float(i) + 40 for i in range(1, 60)], ("a", "b"))
