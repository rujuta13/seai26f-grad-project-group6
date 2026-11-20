"""The harness: asks parsed, scorers run, prompts differ only where treatments differ."""
import os

import pytest

from harness import contracts as c
from harness import run
from harness import score
from harness.loc import count

DUMMY = os.path.join(os.path.dirname(__file__), "dummy")
SPLITS = os.path.exists(os.path.join(run.data.OUT, "ssw-train.csv"))


@pytest.mark.parametrize("domain", c.DOMAIN_ORDER)
def test_twenty_asks_per_domain(domain):
    a = c.asks(domain)
    assert sorted(a) == sorted(c.SKILLS) and all(len(v) == 2 for v in a.values())


@pytest.mark.skipif(not SPLITS, reason="run `make splits` first")
@pytest.mark.parametrize("n", range(1, 11))
def test_scorers_run_on_naive_code(n):
    s = score.score("ssw", n, DUMMY)
    assert s["ok"], s


@pytest.mark.skipif(not SPLITS, reason="run `make splits` first")
def test_crashing_code_is_a_result_not_an_error(tmp_path):
    (tmp_path / "step2.py").write_text("def guess(train_csv, row):\n    raise ValueError('boom')\n")
    s = score.score("ssw", 2, str(tmp_path))
    assert not s["ok"] and "boom" in s["extra"]["error"]


@pytest.mark.skipif(not SPLITS, reason="run `make splits` first")
@pytest.mark.parametrize("pair", [(1, 2), (3, 4), (5, 6)])
def test_twins_differ_only_by_the_reuse_line(pair):
    earlier = [(1, "x = 1"), (2, "y = 2")]
    a, b = (run.prompt(t, 3, earlier) for t in pair)
    assert b.replace(run.REUSE + "\n\n", "") == a


@pytest.mark.skipif(not SPLITS, reason="run `make splits` first")
def test_libraries_differ_between_pairs():
    p1, p3, p5 = (run.prompt(t, 1, []) for t in (1, 3, 5))
    assert "scikit-learn" in p3 and "scikit-learn" not in p1
    assert "# ezr.py" in p5 and "# ezr.py" not in p3


def test_extract_takes_the_python_block():
    assert run.extract("Sure!\n```python\nx = 1\n```\nbye") == "x = 1\n"
    assert run.extract("no code here") == ""


def test_count_skips_blanks_and_comments():
    assert count("# hi\n\nx = 1\n  # more\ny = 2\n") == 2
