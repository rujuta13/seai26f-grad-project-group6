"""Is a gap real, or noise? ezr's test: Cohen, Kolmogorov-Smirnov and Cliff's delta must all
agree before two samples count as the same. Lists in must be sorted."""
from statistics import mean, stdev

TINY = 1e-32


def cohen(xs, ys):
    """Mean gap, in pooled-sd units."""
    n, m = len(xs), len(ys)
    sd = lambda v: stdev(v) if len(v) > 1 else 0
    pooled = (((n - 1) * sd(xs) ** 2 + (m - 1) * sd(ys) ** 2) / (n + m - 2)) ** 0.5
    return abs(mean(xs) - mean(ys)) / (pooled + TINY)


def ks(xs, ys):
    """Biggest gap between the two cdfs, in critical units."""
    nx, ny, d, p, q = len(xs), len(ys), 0, 0, 0
    while p < nx and q < ny:
        v = min(xs[p], ys[q])
        while p < nx and xs[p] == v:
            p += 1
        while q < ny and ys[q] == v:
            q += 1
        d = max(d, abs(p / nx - q / ny))
    return d / ((nx + ny) / (nx * ny)) ** 0.5


def cliffs(xs, ys):
    """Rank imbalance, 0 to 1."""
    gt = lt = j = k = 0
    for x in xs:
        while j < len(ys) and ys[j] < x:
            j += 1
            k = j
        while k < len(ys) and ys[k] == x:
            k += 1
        gt += j
        lt += len(ys) - k
    return abs(gt - lt) / (len(xs) * len(ys))


def same(xs, ys, small=0.35, crit=1.36, delta=0.195):
    """True when all three tests agree the two samples are not different."""
    return cohen(xs, ys) <= small and cliffs(xs, ys) <= delta and ks(xs, ys) <= crit
