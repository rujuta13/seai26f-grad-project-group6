import os
import random

import pytest

from domains.marketing import step21, step29
from spot import changed, fences, glitch, outside
from guess import fit
from table import Table


@pytest.fixture(scope="module")
def t():
    if not os.path.exists(step21.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step21.DATA, step21.SKIP)


def test_typo_changes_one_input_and_skips_no_ops(t):
    r = t.rows[0]
    age = next(c for c in t.cols if c.name == "AGE").at
    assert step29.typo(t, r, "AGE", lambda v: v * 10).cells[age] == r.cells[age] * 10
    assert step29.typo(t, r, "AGE", lambda v: v) is None


def test_impossible_ages_and_incomes_are_flagged_and_clean_ones_are_not(t):
    rng = random.Random(1)
    rows = rng.sample(t.rows, len(t.rows))
    fence = fences(t, rows[:1700], ("AGE", "INCOME"), 1.5)
    test = rows[1700:]
    assert sum(bool(outside(t, r, fence)) for r in test) / len(test) < 0.02
    for name, make in (("AGE", lambda v: 0.0), ("AGE", lambda v: v * 10), ("INCOME", lambda v: v * 10)):
        bad = [m for m in (step29.typo(t, r, name, make) for r in test) if m]
        assert sum(bool(outside(t, m, fence)) for m in bad) / len(bad) > 0.9


def test_floor_stops_flat_count_leaves_flagging_everything(t):
    rng = random.Random(1)
    rows = rng.sample(t.rows, len(t.rows))
    tree, test = fit(t, rows[:1700], rng), rows[1700:][:300]
    loose = sum(bool(glitch(t, tree, r)) for r in test)
    firm = sum(bool(glitch(t, tree, r, floor=2)) for r in test)
    assert firm < loose and firm / len(test) < 0.08


def test_demo_runs(capsys):
    step29.main(step21.DATA, 1, 50)
    out = capsys.readouterr().out
    assert "ask 17:" in out and "ask 18:" in out
