"""step26, justify (marketing): reasons a sales director can check, and reasons we can act on.

    PYTHONPATH=src python3 -m domains.marketing.step26

Same machinery as SS-W step6 and players step16. Ask 11 explains why the customers our step 22
guesser ranks most likely to respond should get the next campaign; ask 12 gives reasons for a low
rating, leaving out anything nobody can change (age, income, household, marital status, education).
"""
import random
import statistics
import sys

from blame import blame
from certify import gap
from domains.marketing.step21 import DATA, SKIP
from guess import fit, guess
from justify import one_line
from remember import spread, traits
from table import Table

FIXED = ("AGE", "INCOME", "KIDHOME", "TEENHOME", "MARITAL_DIVORCED", "MARITAL_MARRIED", "MARITAL_SINGLE",
         "MARITAL_TOGETHER", "MARITAL_WIDOW", "EDUCATION_2N CYCLE", "EDUCATION_BASIC", "EDUCATION_GRADUATION",
         "EDUCATION_MASTER", "EDUCATION_PHD", "CUSTOMER_DAYS+")


def main(path=DATA, seed=1):
    rng, t = random.Random(seed), Table.read(path, SKIP)
    i = [c.name for c in t.y].index("RESPONSE+")
    rows = rng.sample(t.rows, len(t.rows))
    train, test = rows[:len(rows) * 4 // 5], rows[len(rows) * 4 // 5:]
    tree = fit(t, train, rng)
    group = sorted(test, key=lambda r: -guess(t, tree, r)[i][0])[:len(test) // 5]
    rate = lambda rs: sum(r.cells[t.y[i].at] for r in rs) / len(rs)
    print(f"ask 11: why mail the {len(group)} customers we guess are likeliest to respond (of {len(test)} unseen)")
    print(f"  {rate(group):.0%} of them answered the last campaign, against {rate(test):.0%} of all; "
          f"`same` calls that gap {'noise' if gap(t, group, rng.sample(test, 100))[i][2] else 'real'}")
    print("  what sets them apart: " + traits(t, group, spread(t), k=4))
    ranked = sorted(t.rows, key=t.disty)
    k = len(ranked) // 10
    found = blame(t, ranked[-k:], ranked[:k], rng)
    median = statistics.median(map(t.disty, t.rows))
    print("\nask 12: four customers rated low, with reasons we can act on")
    for r in [r for r in rng.sample(t.rows, 40) if t.disty(r) > median][:4]:
        print("  " + one_line(t, r, found, median, skip=FIXED))


if __name__ == "__main__":
    main(*sys.argv[1:2])
