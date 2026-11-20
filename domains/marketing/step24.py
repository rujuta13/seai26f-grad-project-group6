"""step24, flow (marketing): judge new sign-ups as they arrive one at a time.

    PYTHONPATH=src python3 -m domains.marketing.step24

The csv has no sign-up dates, so arrival order is random, and the drift is simulated: after a
change point every new customer's deal purchases are made 3%, 8% or 15% lower. The machinery is
SS-W step4's. The table is small, so each run is a stream of 1,705 customers, not 3,000.
"""
import sys

from domains.marketing.step21 import DATA, SKIP
from flow import report
from table import Table


def main(path=DATA, seed=1, runs=8):
    report(Table.read(path, SKIP), seed, runs, "customers")


if __name__ == "__main__":
    main(*sys.argv[1:2])
