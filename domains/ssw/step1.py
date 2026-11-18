"""step1, remember: the families LLVM flag sets fall into, and the unusually fast, light ones.

    PYTHONPATH=src python3 -m domains.ssw.step1
"""
import random
import sys

from cluster import families
from table import Table

DATA = "data/SS-W.csv"


def on(t, rows, share=0.8):
    """Flags switched on in at least `share` of these rows."""
    return [c.name for c in t.x if sum(r.cells[c.at] == "1" for r in rows) >= share * len(rows)]


def goals(t, rows):
    """Mean of each goal over these rows."""
    return [sum(r.cells[c.at] for r in rows) / len(rows) for c in t.y]


def main(path=DATA, seed=1):
    t = Table.read(path)
    best = sorted(t.rows, key=t.disty)
    print(f"{len(t.rows)} builds, {len(t.x)} flags, goals: {', '.join(c.name for c in t.y)}\n")
    print("families of flag sets (best first):")
    for g in sorted(families(t, random.Random(seed)), key=lambda g: sum(map(t.disty, g)) / len(g)):
        print(f"  {len(g):6d} builds  goals {[round(v, 1) for v in goals(t, g)]}"
              f"  always on: {', '.join(on(t, g)) or '-'}")
    top = best[:len(best) // 100]
    print(f"\nunusually fast and light: the best 1% ({len(top)} builds), goals "
          f"{[round(v, 1) for v in goals(t, top)]} vs {[round(v, 1) for v in goals(t, t.rows)]} overall")
    for r in best[:5]:
        print(f"  {[r.cells[c.at] for c in t.y]}  on: {', '.join(c.name for c in t.x if r.cells[c.at] == '1')}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
