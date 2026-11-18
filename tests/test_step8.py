import os
import random

import pytest

from blame import blame, split
from domains.ssw import step1
from fix import index, plan
from guess import fit
from table import Table


@pytest.fixture(scope="module")
def setup():
    if not os.path.exists(step1.DATA):
        pytest.skip("run `make data` first")
    rng, t = random.Random(1), Table.read(step1.DATA)
    tree, speed = fit(t, t.rows, rng), t.y[0].at
    order = sorted(float(r.cells[speed]) for r in t.rows)
    return t, tree, index(t), blame(t, *split(t, t.rows, t.y[0]), rng), order[len(order) // 4], order[3 * len(order) // 4]


def test_plan_flips_only_what_it_says_and_respects_frozen(setup):
    t, tree, where, found, fast, slow = setup
    speed = t.y[0].at
    slow_ones = [r for r in t.rows if float(r.cells[speed]) >= slow][:200]
    made = [(r, plan(t, tree, where, r, found, fast)) for r in slow_ones]
    made = [(r, g) for r, g in made if g]
    assert len(made) > 100
    for r, (flips, new) in made[:50]:
        changed = [c.name for c in t.x if r.cells[c.at] != new.cells[c.at]]
        assert sorted(changed) == sorted(flips)
    for r in slow_ones:
        got = plan(t, tree, where, r, found, fast, frozen=("inline",))
        assert not got or "inline" not in got[0]


def test_the_best_build_has_nothing_left_to_fix(setup):
    t, tree, where, found, fast, slow = setup
    best = min(t.rows, key=lambda r: float(r.cells[t.y[0].at]))
    assert plan(t, tree, where, best, found, fast) is None
