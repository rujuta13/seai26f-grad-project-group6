import os

import pytest

from domains.marketing import step21, step30


@pytest.mark.skipif(not os.path.exists(step21.DATA), reason="run `make data` first")
def test_the_race_reports_both_asks_and_every_rival(capsys):
    step30.main(step21.DATA, 2, 1500)
    out = capsys.readouterr().out
    for word in ("ask 19:", "ask 20:", "top income", "forest", "random", "did not respond"):
        assert word in out
    rates = [float(l.split("%")[0].split()[-1]) for l in out.splitlines() if "did not respond" in l and "ask" not in l]
    assert rates[0] < rates[2] and rates[1] < rates[2]  # both models beat random mailing
