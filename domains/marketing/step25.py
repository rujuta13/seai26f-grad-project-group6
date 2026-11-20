"""step25, blame (marketing): why customers never respond, and what complainers share.

    PYTHONPATH=src python3 -m domains.marketing.step25

Same machinery as SS-W step5 and players step15. The campaign columns define the groups in ask 9,
so they are left out of its list. Only 20 customers complain, so ask 10 has little to go on.
"""
import random
import sys

from blame import blame, show
from domains.marketing.step21 import DATA, SKIP
from table import Table


def main(path=DATA, seed=1):
    rng, t = random.Random(seed), Table.read(path, SKIP)
    at = {c.name: c.at for c in t.cols}
    camps = ["ACCEPTEDCMP1", "ACCEPTEDCMP2", "ACCEPTEDCMP3", "ACCEPTEDCMP4", "ACCEPTEDCMP5", "RESPONSE+"]
    said = lambda r: sum(r.cells[at[c]] for c in camps)
    never, often = [r for r in t.rows if said(r) == 0], [r for r in t.rows if said(r) >= 2]
    found = [f for f in blame(t, never, often, rng) if not f[0].startswith("ACCEPTEDCMP")]
    print(f"ask 9: customers who answered none of the six campaigns ({len(never)}) vs two or more ({len(often)})")
    show(t, found, never, often, ("never", "often"))
    yes = [r for r in t.rows if r.cells[at["COMPLAIN-"]] == 1]
    rest = [r for r in t.rows if r.cells[at["COMPLAIN-"]] == 0]
    print(f"\nask 10: customers who complained ({len(yes)}) vs a sample of those who did not")
    show(t, blame(t, yes, rng.sample(rest, 100), rng), yes, rest, ("complained", "others"))


if __name__ == "__main__":
    main(*sys.argv[1:2])
