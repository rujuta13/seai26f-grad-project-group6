import os
import random

import pytest

from blame import blame, split
from domains.players import step11
from domains.players.step18 import FROZEN
from fix import improve
from guess import near
from table import Table


@pytest.fixture(scope="module")
def setup():
    if not os.path.exists(step11.DATA):
        pytest.skip("run `make data` first")
    t = Table.read(step11.DATA, step11.SKIP)
    rows = random.Random(1).sample(t.rows, len(t.rows))
    found = blame(t, *split(t, rows, t.y[0]), random.Random(1))
    return t, rows, found


def test_plan_moves_only_trainable_columns_to_the_good_median(setup):
    t, rows, found = setup
    good = sorted(rows, key=lambda r: -r.cells[t.y[0].at])[:2000]
    weak = [r for r in rows if r.cells[t.y[0].at] < 40][:50]
    at = {c.name: c.at for c in t.x}
    plans = [improve(t, r, found, good, FROZEN) for r in weak]
    assert sum(p is not None for p in plans) > 40
    for r, (changes, new) in zip(weak, [p for p in plans if p]):
        assert 1 <= len(changes) <= 3
        for name, old, to in changes:
            assert name not in FROZEN and old == r.cells[at[name]] and new.cells[at[name]] == to
            med = sorted(g.cells[at[name]] for g in good)[len(good) // 2]
            assert to == med
        assert new.cells[at["Age"]] == r.cells[at["Age"]]  # untrainable columns stay put


def test_nothing_to_fix_when_the_player_is_the_good_median_itself(setup):
    t, rows, found = setup
    good = [rows[0]] * 5
    assert improve(t, rows[0], found, good, FROZEN) is None


def test_near_guesses_a_strong_players_penalties_high(setup):
    t, rows, _ = setup
    ref = rows[:3000]
    best = max(rows[3000:], key=lambda r: r.cells[t.y[0].at])
    worst = min(rows[3000:], key=lambda r: r.cells[t.y[0].at])
    assert near(t, ref, best) > near(t, ref, worst)
