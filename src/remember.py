"""Remember: the families rows fall into, and the unusually good rows."""
import random

from cluster import families


def goals(t, rows):
    """Mean of each goal over these rows."""
    return [sum(r.cells[c.at] for r in rows) / len(rows) for c in t.y]


def spread(t):
    """(mean, standard deviation) of every numeric input column."""
    out = {}
    for c in t.x:
        if c.num:
            v = [r.cells[c.at] for r in t.rows if r.cells[c.at] != "?"]
            m = sum(v) / len(v)
            out[c.at] = (m, (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5 or 1)
    return out


def traits(t, rows, spec, k=3):
    """The `k` numeric columns on which these rows differ most from everyone, in standard deviations."""
    z = [((sum(r.cells[c.at] for r in rows) / len(rows) - spec[c.at][0]) / spec[c.at][1], c.name)
         for c in t.x if c.at in spec]
    return ", ".join(f"{'high' if d > 0 else 'low'} {n}" for d, n in sorted(z, key=lambda p: -abs(p[0]))[:k])


def report(t, seed, describe, show, things, inputs):
    """Print the families (best first), then the best 1% and its five best rows."""
    best = sorted(t.rows, key=t.disty)
    print(f"{len(t.rows)} {things}, {len(t.x)} {inputs}, goals: {', '.join(c.name for c in t.y)}\n")
    print(f"families of {things} (best first):")
    for g in sorted(families(t, random.Random(seed)), key=lambda g: sum(map(t.disty, g)) / len(g)):
        print(f"  {len(g):6d} {things}  goals {[round(v, 1) for v in goals(t, g)]}  {describe(t, g)}")
    top = best[:len(best) // 100]
    print(f"\nunusually good: the best 1% ({len(top)} {things}), goals "
          f"{[round(v, 1) for v in goals(t, top)]} vs {[round(v, 1) for v in goals(t, t.rows)]} overall")
    for r in best[:5]:
        print(f"  {show(t, r)}")
