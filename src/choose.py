"""Choose: find a very good row while looking at (labelling) very few."""
from cluster import leaf
from guess import fit


def only(t, i):
    """A score for one goal alone (0 is best)."""
    return lambda r: abs(t.y[i].heaven - r.y[i])


def pick(t, rng, budget=30, score=None, start=6, pool=100, rows=None):
    """Look at `start` random rows, then `budget - start` more. Each time, cluster the rows seen so
    far, guess each of `pool` random candidates by the mean score of its leaf, and look at the best
    guess. Returns the best row seen."""
    score, rows = score or t.disty, rows or t.rows
    seen = {id(r): (r, score(r)) for r in rng.sample(rows, start)}
    while len(seen) < budget:
        tree = fit(t, [r for r, _ in seen.values()], rng, stop=4)
        def mean(r):
            s = [seen[id(x)][1] for x in leaf(t, tree, r).rows]
            return sum(s) / len(s)
        best = min((r for r in rng.sample(rows, pool) if id(r) not in seen), key=mean)
        seen[id(best)] = (best, score(best))
    return min(seen.values(), key=lambda p: p[1])[0]


def compare(t, budget, repeats, things, rows=None, score=None):
    """Print how good, as a rank among all `rows`, the best row found with `budget` looks is: our
    tuner against picking at random with the same budget (rank 1% = better than 99% of rows)."""
    import bisect
    import random
    import statistics
    rows, score = rows or t.rows, score or t.disty
    ranked = sorted(map(score, rows))
    pct = lambda r: 100 * bisect.bisect_left(ranked, score(r)) / len(ranked)
    for name, find in (("tuner ", lambda rng: pick(t, rng, budget, score, rows=rows)),
                       ("random", lambda rng: min(rng.sample(rows, budget), key=score))):
        got = [pct(find(random.Random(s))) for s in range(repeats)]
        print(f"  {name} median rank {statistics.median(got):.1f}%, worst of {repeats} runs {max(got):.1f}%")
