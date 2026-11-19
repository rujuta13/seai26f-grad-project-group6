"""step22, guess (marketing): guess a customer's buying channels and whether they will answer a campaign.

    PYTHONPATH=src python3 -m domains.marketing.step22

Same machinery as SS-W step2 and players step12. Response is a 0/1 goal, so its guess is a
probability and its "10-90% range" is always 0 to 1; the extra lines below judge it as a ranking.
"""
import sys

from domains.marketing.step21 import DATA, SKIP
from guess import report
from table import Table


def main(path=DATA, seed=1):
    t = Table.read(path, SKIP)
    at = {c.name: c.at for c in t.cols}
    test, out = report(t, seed, "customers",
                       lambda t, r: f"customer, income {r.cells[at['INCOME']]:.0f}, age {r.cells[at['AGE']]:.0f}")
    i = [c.name for c in t.y].index("RESPONSE+")
    ranked = sorted(zip(out, test), key=lambda p: -p[0][i][0])
    top = ranked[:len(ranked) // 5]
    rate = lambda rs: sum(r.cells[at["RESPONSE+"]] for r in rs) / len(rs)
    print(f"\nask 4: answered the last campaign: {rate(test):.0%} of all {len(test)} unseen customers, "
          f"{rate([r for _, r in top]):.0%} of the {len(top)} we guessed most likely to")


if __name__ == "__main__":
    main(*sys.argv[1:2])
