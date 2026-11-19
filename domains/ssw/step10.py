"""step10, race: our tuner and guesser against off-the-shelf rivals.

    PYTHONPATH=src python3 -m domains.ssw.step10

Rivals: random search, and a random-forest tuner and guesser from scikit-learn. Every contender
gets the same budget. `same` (step 3) says whether a difference between two is real.
"""
import sys

from race import guess_race, tuner_race
from table import Table

DATA = "data/SS-W.csv"


def main(path=DATA, budget=30, repeats=40, ntrain=500):
    t = Table.read(path)
    print("ask 19: " + "", end="")
    tuner_race(t, budget, repeats, "builds")
    print("\nask 20: ", end="")
    guess_race(t, ntrain, 10, "builds")


if __name__ == "__main__":
    main(*sys.argv[1:2])
