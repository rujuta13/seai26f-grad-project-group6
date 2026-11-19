import os
import random

import pytest

from cluster import families
from domains.players import step11
from remember import spread, traits
from table import Table


def test_columns_that_never_vary_or_are_named_are_not_inputs(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("id,Age,Const,Size-\n1,10,5,1\n2,20,5,2\n3,30,5,3\n")
    t = Table.read(p, skip=("id",))
    assert [c.name for c in t.x] == ["Age"] and len(t.rows[0].x) == 1


@pytest.fixture(scope="module")
def players():
    if not os.path.exists(step11.DATA):
        pytest.skip("run `make data` first")
    return Table.read(step11.DATA, step11.SKIP)


def test_player_inputs_exclude_ids_and_blank_columns(players):
    names = {c.name for c in players.x}
    assert not names & {"num", "Rank", "Name", "url", "Position", "League", "Team"}
    assert {"Finishing", "Age", "Reactions"} <= names and [c.name for c in players.y] == ["Penalties+", "Strength+"]


def test_families_of_players_differ_in_a_readable_way(players):
    gs = families(players, random.Random(1))
    spec = spread(players)
    assert 4 <= len(gs) <= 16 and sum(map(len, gs)) == len(players.rows)
    said = {traits(players, g, spec) for g in gs}
    assert len(said) == len(gs) and all(s.count(",") == 2 for s in said)
