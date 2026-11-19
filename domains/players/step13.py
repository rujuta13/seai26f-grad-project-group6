"""step13, certify (players): is a gap between players real, or luck?

    PYTHONPATH=src python3 -m domains.players.step13
"""
import random
import sys

from certify import gap, say
from cluster import families
from domains.players.step11 import DATA, SKIP
from guess import fit, guess
from table import Table


def predicted(t, tree, row):
    """How good a player is guessed to be, before seeing his goals: guess's means, as distance to heaven."""
    m = guess(t, tree, row)
    return sum((c.heaven - c.norm(g[0])) ** 2 for c, g in zip(t.y, m)) ** 0.5


def main(path=DATA, seed=1, n=100):
    rng, t = random.Random(seed), Table.read(path, SKIP)
    rows = rng.sample(t.rows, len(t.rows))
    train, test = rows[:len(rows) * 4 // 5], rows[len(rows) * 4 // 5:]
    tree = fit(t, train, rng)
    shortlist = sorted(test, key=lambda r: predicted(t, tree, r))[:50]
    print("ask 5: our shortlist (50 players picked from their other ratings, goals unseen) vs 100 random players")
    for c, g in zip(t.y, gap(t, shortlist, rng.sample(test, n))):
        print(f"  {c.name:12} {say(g, 'shortlist')}")
    at = {c.name: c.at for c in t.cols}
    mean = lambda g, name: sum(r.cells[at[name]] for r in g) / len(g)
    gs = families(t, random.Random(seed))
    att, dfn = max(gs, key=lambda g: mean(g, "SHO")), max(gs, key=lambda g: mean(g, "DEF"))
    print(f"\nask 6: the attacking type ({len(att)} players, highest shooting) vs the defending type "
          f"({len(dfn)} players, highest defending), {n} each")
    for c, g in zip(t.y, gap(t, rng.sample(att, n), rng.sample(dfn, n))):
        print(f"  {c.name:12} {say(g, 'attacking type')}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
