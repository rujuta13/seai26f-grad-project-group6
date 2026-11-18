import os
import random

import pytest

from cluster import families
from domains.ssw import step1, step9
from guess import fit
from spot import cutoff, glitch, odd
from table import Table


@pytest.fixture(scope="module")
def ssw():
    if not os.path.exists(step1.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step1.DATA)


def test_glitches_are_flagged_and_clean_results_mostly_are_not(ssw):
    rng = random.Random(1)
    rows = rng.sample(ssw.rows, 4000)
    tree = fit(ssw, rows[:3200], rng)
    test = rows[3200:3500]
    assert sum(bool(glitch(ssw, tree, r)) for r in test) / len(test) < 0.05
    assert sum(bool(glitch(ssw, tree, step9.scaled(ssw, r, 4))) for r in test) / len(test) > 0.9
    assert "Performance-" in glitch(ssw, tree, step9.scaled(ssw, test[0], 4))


def test_other_families_are_odd_and_the_same_family_is_not(ssw):
    gs = sorted(families(ssw, random.Random(1)), key=lambda g: sum(map(ssw.disty, g)) / len(g))
    rng = random.Random(2)
    seen = rng.sample(gs[0], 200)
    cut = cutoff(ssw, seen)
    same = sum(odd(ssw, seen, r, cut) for r in rng.sample(gs[0], 100))
    other = sum(odd(ssw, seen, r, cut) for r in rng.sample(gs[1], 100))
    assert same < 20 and other > 40
