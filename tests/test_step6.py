import os
import random
import statistics

import pytest

from blame import blame, split
from certify import gap
from domains.ssw import step1
from justify import one_line, recommend
from table import Table


@pytest.fixture(scope="module")
def ssw():
    if not os.path.exists(step1.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step1.DATA)


def test_recommendation_names_the_flags_step3_found_and_beats_random(ssw):
    rng = random.Random(1)
    rec = recommend(ssw, rng)
    said = {n: (s, why) for n, s, why in rec}
    assert said["inline"][0] == "off" and "better" in said["inline"][1]
    assert "no real effect" in said["mergereturn"][1]
    chosen = [r for r in ssw.rows if all(r.cells[c.at] == ("1" if s == "on" else "0")
                                         for c, (_, s, why) in zip(ssw.x, rec) if "no real" not in why)]
    (a, b, noise), _ = gap(ssw, rng.sample(chosen, 100), rng.sample(ssw.rows, 100))
    assert not noise and a < b


def test_one_line_gives_a_verdict_and_reasons(ssw):
    rng = random.Random(2)
    found = blame(ssw, *split(ssw, ssw.rows, ssw.y[0]), rng)
    median = statistics.median(map(ssw.disty, ssw.rows))
    bad = max(ssw.rows, key=ssw.disty)
    good = min(ssw.rows, key=ssw.disty)
    assert one_line(ssw, bad, found, median).startswith("Rated bad: ")
    assert one_line(ssw, good, found, median).startswith("Rated good: ")
    assert "is on" in one_line(ssw, bad, found, median) or "is off" in one_line(ssw, bad, found, median)
