"""step20, race (players): our shortlist and guesser against simple and off-the-shelf rivals.

    PYTHONPATH=src python3 -m domains.players.step20

Ask 19 builds 50-player shortlists from unseen players four ways and scores each by the players'
real penalties and strength (distance to heaven, lower is better). Ask 20 is the same guesser
race as SS-W step10, on penalties.
"""
import random
import statistics
import sys

from domains.players.step11 import DATA, SKIP
from domains.players.step13 import predicted
from guess import fit
from race import forest_guess, guess_race, verdict
from table import Table


def main(path=DATA, splits=8, ntrain=4000, ntest=3000):
    t = Table.read(path, SKIP)
    ovr = next(c.at for c in t.cols if c.name == "OVR")
    score = {k: [] for k in ("ours", "forest", "top OVR", "random")}
    for s in range(splits):
        rng = random.Random(s)
        rows = rng.sample(t.rows, ntrain + ntest)
        train, test = rows[:ntrain], rows[ntrain:]
        tree = fit(t, train, rng)
        guesses = [forest_guess(t, train, test, g) for g in (0, 1)]
        want = lambda i: sum((c.heaven - c.norm(guesses[k][i])) ** 2 for k, c in enumerate(t.y)) ** 0.5
        lists = {"ours": sorted(test, key=lambda r: predicted(t, tree, r)),
                 "forest": [test[i] for i in sorted(range(len(test)), key=want)],
                 "top OVR": sorted(test, key=lambda r: -r.cells[ovr]),
                 "random": test}
        for k, rs in lists.items():
            score[k].append(statistics.fmean(map(t.disty, rs[:50])))
    print(f"ask 19: four 50-player shortlists from {ntest} unseen players, {splits} repeats (mean distance to heaven; lower is better)")
    for k, v in score.items():
        print(f"  {k:8} {statistics.median(v):.3f}")
    for rival in ("top OVR", "forest", "random"):
        print("  " + verdict(score["ours"], score[rival], ("ours", rival)))
    print("\nask 20: ", end="")
    guess_race(t, 1000, 10, "players")


if __name__ == "__main__":
    main(*sys.argv[1:2])
