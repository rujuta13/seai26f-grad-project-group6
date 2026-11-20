"""step30, race (marketing): our targeting and response guesser against simple and off-the-shelf rivals.

    PYTHONPATH=src python3 -m domains.marketing.step30

Ask 19 builds 50-customer mailing lists from unseen customers four ways (our guesser over every goal, a
random forest over every goal, "highest income first", random) and scores each by the customers' real goals
(distance to heaven, lower is better). Ask 20 mails the 20% of unseen customers each model thinks most likely to
respond and scores the share who really did (shown as the miss rate, so lower is better like everywhere else).
"""
import random
import statistics
import sys

from domains.marketing.step21 import DATA, SKIP
from domains.players.step13 import predicted
from guess import fit, guess
from race import forest_guess, verdict
from table import Table


def main(path=DATA, splits=10, ntrain=1500):
    t = Table.read(path, SKIP)
    income = next(c.at for c in t.cols if c.name == "INCOME")
    resp = [c.name for c in t.y].index("RESPONSE+")
    at, ntest = t.y[resp].at, len(t.rows) - ntrain
    score = {k: [] for k in ("ours", "forest", "top income", "random")}
    miss = {k: [] for k in ("ours", "forest", "random")}
    for s in range(splits):
        rng = random.Random(s)
        rows = rng.sample(t.rows, len(t.rows))
        train, test = rows[:ntrain], rows[ntrain:]
        tree = fit(t, train, rng)
        guesses = [forest_guess(t, train, test, g) for g in range(len(t.y))]
        want = lambda i: sum((c.heaven - c.norm(guesses[k][i])) ** 2 for k, c in enumerate(t.y)) ** 0.5
        lists = {"ours": sorted(test, key=lambda r: predicted(t, tree, r)),
                 "forest": [test[i] for i in sorted(range(len(test)), key=want)],
                 "top income": sorted(test, key=lambda r: -r.cells[income]),
                 "random": test}
        for k, rs in lists.items():
            score[k].append(statistics.fmean(map(t.disty, rs[:50])))
        top = ntest // 5
        mine = sorted(test, key=lambda r: -guess(t, tree, r)[resp][0])[:top]
        theirs = [test[i] for i in sorted(range(ntest), key=lambda i: -guesses[resp][i])[:top]]
        for k, rs in (("ours", mine), ("forest", theirs), ("random", test[:top])):
            miss[k].append(1 - statistics.fmean(r.cells[at] for r in rs))
    print(f"ask 19: four 50-customer mailing lists from {ntest} unseen customers, {splits} repeats (mean distance to heaven; lower is better)")
    for k, v in score.items():
        print(f"  {k:10} {statistics.median(v):.3f}")
    for rival in ("top income", "forest", "random"):
        print("  " + verdict(score["ours"], score[rival], ("ours", rival)))
    print(f"\nask 20: mail the 20% of {ntest} unseen customers each model rates likeliest to respond; share who did NOT respond (lower is better)")
    for k, v in miss.items():
        print(f"  {k:10} {statistics.median(v):.1%} did not respond")
    for rival in ("forest", "random"):
        print("  " + verdict(miss["ours"], miss[rival], ("ours", rival)))


if __name__ == "__main__":
    main(*sys.argv[1:2])
