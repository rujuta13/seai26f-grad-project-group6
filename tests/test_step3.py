import os
import random

import pytest

from certify import gap
from cluster import families
from domains.ssw import step1
from stats import cliffs, cohen, ks, same
from table import Table


def test_known_values():
    xs, ys = [1, 2, 3], [4, 5, 6]
    assert cliffs(xs, ys) == 1 and cliffs(xs, xs) == 0
    assert ks(xs, xs) == 0 and ks(xs, ys) > ks(xs, [3, 4, 5])
    assert cohen(xs, xs) == 0 and abs(cohen(xs, ys) - 3) < 1e-9  # gap 3, pooled sd 1


def test_same_says_yes_for_shifted_noise_and_no_for_a_real_gap():
    rng = random.Random(1)
    a, b = ([rng.gauss(0, 1) for _ in range(100)] for _ in range(2))
    assert same(sorted(a), sorted(b))
    assert not same(sorted(a), sorted(x + 2 for x in b))


@pytest.fixture(scope="module")
def ssw():
    if not os.path.exists(step1.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step1.DATA)


def test_gap_between_best_and_worst_family(ssw):
    rng = random.Random(1)
    gs = sorted(families(ssw, rng), key=lambda g: sum(map(ssw.disty, g)) / len(g))
    best, worst = rng.sample(gs[0], 100), rng.sample(gs[-1], 100)
    (m1, m2, noise), _ = gap(ssw, best, worst)
    assert not noise and m1 < m2  # real, and the best family is faster


def test_gap_between_two_halves_of_one_family_is_noise(ssw):
    rng = random.Random(1)
    rows = rng.sample(families(ssw, rng)[0], 200)
    assert all(g[2] for g in gap(ssw, rows[:100], rows[100:]))
