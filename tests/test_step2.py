import os
import random

import pytest

from domains.ssw import step1
from guess import fit, guess
from table import Table

CSV = "Age,Size-,Gain+\n10,1,9\n20,3,5\n30,2,7\n40,4,3\n"


def test_one_leaf_guesses_the_mean(tmp_path):
    p = tmp_path / "tiny.csv"
    p.write_text(CSV)
    t = Table.read(p)
    tree = fit(t, t.rows, random.Random(1), stop=100)  # stop is larger than the table: one leaf
    (m1, lo1, hi1), (m2, lo2, hi2) = guess(t, tree, t.rows[0])
    assert (m1, m2) == (2.5, 6.0) and lo1 <= hi1 and lo2 <= hi2


def test_guesses_beat_the_average_on_unseen_builds():
    if not os.path.exists(step1.DATA):
        pytest.skip("run `make data` first")
    rng, t = random.Random(1), Table.read(step1.DATA)
    rows = rng.sample(t.rows, 20000)
    train, test = rows[:16000], rows[16000:]
    tree = fit(t, train, rng)
    out = [guess(t, tree, r) for r in test]
    for i, c in enumerate(t.y):
        mean = sum(r.cells[c.at] for r in train) / len(train)
        ours = sum(abs(g[i][0] - r.cells[c.at]) for g, r in zip(out, test))
        base = sum(abs(mean - r.cells[c.at]) for r in test)
        assert ours < base  # even CPU, which is hard, must at least not do worse
        if c.name == "Performance-":
            assert ours < 0.85 * base
