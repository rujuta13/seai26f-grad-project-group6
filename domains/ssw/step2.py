"""step2, guess: predict a flag set's speed and CPU before benchmarking it, and say how sure.

    PYTHONPATH=src python3 -m domains.ssw.step2
"""
import random
import sys

from guess import fit, guess
from table import Table

DATA = "data/SS-W.csv"


def main(path=DATA, seed=1, stop=32):
    rng, t = random.Random(seed), Table.read(path)
    rows = rng.sample(t.rows, len(t.rows))
    train, test = rows[:len(rows) * 4 // 5], rows[len(rows) * 4 // 5:]
    tree = fit(t, train, rng, stop)
    out = [guess(t, tree, r) for r in test]
    print(f"trained on {len(train)} builds, guessed {len(test)} unseen ones\n")
    print(f"{'goal':14} {'our error':>10} {'always-the-mean':>16} {'inside 10-90%':>14}")
    for i, c in enumerate(t.y):
        mean = sum(r.cells[c.at] for r in train) / len(train)
        ours = sum(abs(g[i][0] - r.cells[c.at]) for g, r in zip(out, test)) / len(test)
        base = sum(abs(mean - r.cells[c.at]) for r in test) / len(test)
        sure = sum(g[i][1] <= r.cells[c.at] <= g[i][2] for g, r in zip(out, test)) / len(test)
        print(f"{c.name:14} {ours:10.3f} {base:16.3f} {sure:13.0%}")
    print("\nthree examples (guess [low, high] vs actual):")
    for g, r in list(zip(out, test))[:3]:
        print("  on:", ", ".join(c.name for c in t.x if r.cells[c.at] == "1") or "-")
        for (m, lo, hi), c in zip(g, t.y):
            print(f"    {c.name:12} {m:10.2f} [{lo:.2f}, {hi:.2f}]  actual {r.cells[c.at]:.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
