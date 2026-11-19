"""Guess a row's goals from the training rows that fall into the same cluster-tree leaf."""
import random

from cluster import grow, leaf


def fit(t, rows, rng, stop=32):
    return grow(t, rows, rng, stop)


def guess(t, tree, row, lo=0.1, hi=0.9):
    """For each goal, (mean, low, high) over the leaf's rows; [low, high] says how sure."""
    rows, out = leaf(t, tree, row).rows, []
    for c in t.y:
        v = sorted(r.cells[c.at] for r in rows if r.cells[c.at] != "?")
        out.append((sum(v) / len(v), v[int(lo * (len(v) - 1))], v[int(hi * (len(v) - 1))]))
    return out


def near(t, ref, row, k=15, goal=0):
    """Median of a goal over the `k` rows of `ref` closest to `row`: a guess that moves smoothly when
    the row changes a little, unlike a leaf's mean, which jumps when the row crosses a leaf's edge."""
    c = t.y[goal].at
    v = sorted(ref, key=lambda r: t.distx(row, r))[:k]
    return sorted(r.cells[c] for r in v)[k // 2]


def report(t, seed, things, show, stop=32):
    """Train on 80% of the rows, guess the other 20%, and compare with always guessing the mean."""
    rng = random.Random(seed)
    rows = rng.sample(t.rows, len(t.rows))
    train, test = rows[:len(rows) * 4 // 5], rows[len(rows) * 4 // 5:]
    tree = fit(t, train, rng, stop)
    out = [guess(t, tree, r) for r in test]
    print(f"trained on {len(train)} {things}, guessed {len(test)} unseen ones\n")
    print(f"{'goal':14} {'our error':>10} {'always-the-mean':>16} {'inside 10-90%':>14}")
    for i, c in enumerate(t.y):
        mean = sum(r.cells[c.at] for r in train) / len(train)
        ours = sum(abs(g[i][0] - r.cells[c.at]) for g, r in zip(out, test)) / len(test)
        base = sum(abs(mean - r.cells[c.at]) for r in test) / len(test)
        sure = sum(g[i][1] <= r.cells[c.at] <= g[i][2] for g, r in zip(out, test)) / len(test)
        print(f"{c.name:14} {ours:10.3f} {base:16.3f} {sure:13.0%}")
    print("\nthree examples (guess [low, high] vs actual):")
    for g, r in list(zip(out, test))[:3]:
        print("  " + show(t, r))
        for (m, lo, hi), c in zip(g, t.y):
            print(f"    {c.name:12} {m:10.2f} [{lo:.2f}, {hi:.2f}]  actual {r.cells[c.at]:.2f}")
