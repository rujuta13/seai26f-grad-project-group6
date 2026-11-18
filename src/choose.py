"""Choose: find a very good row while looking at (labelling) very few."""
from cluster import leaf
from guess import fit


def only(t, i):
    """A score for one goal alone (0 is best)."""
    return lambda r: abs(t.y[i].heaven - r.y[i])


def pick(t, rng, budget=30, score=None, start=6, pool=100):
    """Look at `start` random rows, then `budget - start` more. Each time, cluster the rows seen so
    far, guess each of `pool` random candidates by the mean score of its leaf, and look at the best
    guess. Returns the best row seen."""
    score = score or t.disty
    seen = {id(r): (r, score(r)) for r in rng.sample(t.rows, start)}
    while len(seen) < budget:
        tree = fit(t, [r for r, _ in seen.values()], rng, stop=4)
        def mean(r):
            s = [seen[id(x)][1] for x in leaf(t, tree, r).rows]
            return sum(s) / len(s)
        best = min((r for r in rng.sample(t.rows, pool) if id(r) not in seen), key=mean)
        seen[id(best)] = (best, score(best))
    return min(seen.values(), key=lambda p: p[1])[0]
