"""Score one step's code against its contract, on the shared split.

    python3 harness/score.py --domain ssw --step 2 --code runs/t2-r1/

Imports <code>/step<N>.py (with <code> on sys.path so earlier steps can be imported),
calls the contract's functions, and prints one JSON line. A crash is a result, not an
error: {"ok": false, "error": "..."}. The runner calls this in a subprocess with a timeout.
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from harness import contracts as c          # noqa: E402
from harness import data                    # noqa: E402


def env_for(domain, tmp):
    splits = data.OUT
    train_csv = os.path.join(splits, f"{domain}-train.csv")
    test_csv = os.path.join(splits, f"{domain}-test.csv")
    header, train = c.read(train_csv)
    _, test = c.read(test_csv)
    _, full = c.read(os.path.join(data.DATA, data.DOMAINS[domain]))
    blind_csv = os.path.join(tmp, f"{domain}-blind.csv")
    c.blank_goals(header, train, blind_csv)
    return dict(header=header, train=train, test=test, full=full, train_csv=train_csv,
                test_csv=test_csv, blind_csv=blind_csv, heaven=c.Heaven(header, full), tmp=tmp)


def load(code_dir, n):
    sys.path.insert(0, os.path.abspath(code_dir))
    spec = importlib.util.spec_from_file_location(f"step{n}", os.path.join(code_dir, f"step{n}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def score(domain, n, code_dir):
    skill = c.step_of(domain, n)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            out = c.SCORE[skill](load(code_dir, n), env_for(domain, tmp))
        except Exception as e:  # the code under test failed: record it, don't crash
            out = c.result(False, error=f"{type(e).__name__}: {e}",
                           trace=traceback.format_exc(limit=3)[-800:])
    return {"domain": domain, "step": n, "skill": skill, **out}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="ssw")
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--code", required=True)
    a = p.parse_args()
    print(json.dumps(score(a.domain, a.step, a.code), default=str))


if __name__ == "__main__":
    main()
