"""step28, fix (marketing): the smallest nudge for a customer who is slipping, and the offer for a deal-only shopper.

    PYTHONPATH=src python3 -m domains.marketing.step28

Same machinery as players step18: take the (at most three) ratings where the customer trails the
good ones most, move each to those customers' median, then check against real customers the plan
never used. Aim: store purchases, the client's usual yardstick. Association, not a promise.
Ask 15 may nudge product spend and recency; ask 16 may only offer a product (spend by product).
"""
import random
import sys

from blame import blame, split
from domains.marketing.step21 import DATA, SKIP
from fix import evaluate
from table import Table


def main(path=DATA, seed=1, trials=60):
    rng, t = random.Random(seed), Table.read(path, SKIP)
    names = [c.name for c in t.x]
    at = {c.name: c.at for c in t.cols}
    store = [c.name for c in t.y].index("NUMSTOREPURCHASES+")
    rows = rng.sample(t.rows, len(t.rows))
    ref, rest = rows[:800], rows[800:]
    found = blame(t, *split(t, rest, t.y[store]), rng)
    v = sorted(r.cells[t.y[store].at] for r in rest)
    want, good = v[int(.75 * len(v))], [r for r in rest if r.cells[t.y[store].at] >= v[int(.75 * len(v))]]
    products = [n for n in names if n.startswith("MNT")]
    nudge = [n for n in names if n not in products + ["RECENCY"]]
    slipping = [r for r in rest if r.cells[at["RECENCY"]] >= 70 and r.cells[t.y[store].at] < v[len(v) // 2]]
    evaluate(t, rng, slipping, ref, found, good, want, nudge, 15, "slipping away (last bought 70+ days ago)", "customers", store, trials)
    deal = [r for r in rest if r.cells[at["NUMDEALSPURCHASES+"]] >= 3 and r.cells[t.y[store].at] <= 3]
    evaluate(t, rng, deal, ref, found, good, want, [n for n in names if n not in products], 16,
             "who only buy on deals (3+ deal purchases, at most 3 store)", "customers", store, trials)


if __name__ == "__main__":
    main(*sys.argv[1:2])
