import os
import random

import pytest

from domains.players import step11, step12
from guess import fit, guess
from table import Table


@pytest.fixture(scope="module")
def players():
    if not os.path.exists(step11.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step11.DATA, step11.SKIP)


def test_guesses_beat_the_average_on_unseen_players_and_ranges_hold(players):
    rng = random.Random(1)
    rows = rng.sample(players.rows, 8000)
    train, test = rows[:6400], rows[6400:]
    tree = fit(players, train, rng)
    out = [guess(players, tree, r) for r in test]
    for i, c in enumerate(players.y):
        mean = sum(r.cells[c.at] for r in train) / len(train)
        ours = sum(abs(g[i][0] - r.cells[c.at]) for g, r in zip(out, test))
        assert ours < 0.7 * sum(abs(mean - r.cells[c.at]) for r in test)
        assert sum(g[i][1] <= r.cells[c.at] <= g[i][2] for g, r in zip(out, test)) / len(test) > 0.65


def test_who_names_a_player(players):
    assert step12.who(players, players.rows[0]).startswith("player with OVR 91")
