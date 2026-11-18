import os
import random

import pytest

from domains.ssw import step1
from race import bits, forest_guess, forest_pick
from table import Table


@pytest.fixture(scope="module")
def ssw():
    if not os.path.exists(step1.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step1.DATA)


def test_forest_tuner_spends_exactly_its_budget(ssw):
    calls = []
    def score(r):
        calls.append(r)
        return ssw.disty(r)
    forest_pick(ssw, random.Random(1), 12, score)
    assert len(calls) == 12


def test_forest_guesser_learns_speed_from_flags(ssw):
    rng = random.Random(1)
    rows = rng.sample(ssw.rows, 700)
    guesses = forest_guess(ssw, rows[:500], rows[500:])
    truth = [r.cells[ssw.y[0].at] for r in rows[500:]]
    mean = sum(r.cells[ssw.y[0].at] for r in rows[:500]) / 500
    err = lambda ps: sum(abs(p - y) for p, y in zip(ps, truth))
    assert len(bits(ssw, rows[:3])[0]) == len(ssw.x) and err(guesses) < err([mean] * 200)
