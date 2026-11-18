import os
import random

import pytest

from cluster import families
from domains.ssw import step1
from table import Table

CSV = "Age,name,Size-,Gain+,noteX\n10,a,1,9,zz\n20,b,3,5,zz\n30,a,2,?,zz\n"


@pytest.fixture
def tiny(tmp_path):
    p = tmp_path / "tiny.csv"
    p.write_text(CSV)
    return Table.read(p)


def test_header_rules(tiny):
    assert [c.name for c in tiny.x] == ["Age", "name"]  # goals and X columns are not inputs
    assert [c.name for c in tiny.y] == ["Size-", "Gain+"]
    assert tiny.cols[0].num and not tiny.cols[1].num
    assert tiny.rows[0].cells[:4] == [10.0, "a", 1.0, 9.0]


def test_distances(tiny):
    a, b, c = tiny.rows
    assert tiny.distx(a, a) == 0 and tiny.distx(a, b) == tiny.distx(b, a) > 0
    assert tiny.distx(a, c) < tiny.distx(a, b)  # a and c share a name, so they sit closer
    assert tiny.disty(a) == 0 and tiny.disty(b) == 1  # a has the best of both goals, b the worst
    assert 0 <= tiny.disty(c) <= 1  # a missing goal is counted as far, not a crash


@pytest.fixture(scope="module")
def ssw():
    if not os.path.exists(step1.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step1.DATA)


def test_families_partition_the_rows(ssw):
    groups = families(ssw, random.Random(1))
    assert sum(map(len, groups)) == len(ssw.rows)
    assert 4 <= len(groups) <= 16  # "a handful"


def test_best_builds_beat_the_rest(ssw):
    d = sorted(map(ssw.disty, ssw.rows))
    assert d[len(d) // 100] < d[len(d) // 2]
    assert all(0 <= x <= 1 for x in d)
