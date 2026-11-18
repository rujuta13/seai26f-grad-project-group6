"""Split rows into similar groups by recursive two-way clustering (cluster, tree, leaf)."""


class Leaf:
    def __init__(self, rows):
        self.rows = rows


class Node:
    def __init__(self, a, b, c, mid, left, right):
        self.a, self.b, self.c, self.mid, self.left, self.right = a, b, c, mid, left, right


def far(t, row, rows, rng, n=30):
    """The row, among n random ones, farthest from `row`."""
    return max(rng.sample(rows, min(n, len(rows))), key=lambda r: t.distx(row, r))


def project(t, a, b, c, row):
    """Where a row falls on the line from pole a to pole b."""
    da, db = t.distx(row, a), t.distx(row, b)
    return (da * da + c * c - db * db) / (2 * c)


def grow(t, rows, rng, stop):
    """A tree that cuts rows in half, by their distance to two far-apart poles, until a leaf
    holds at most `stop` rows."""
    if len(rows) <= stop:
        return Leaf(rows)
    a = far(t, rng.choice(rows), rows, rng)
    b = far(t, a, rows, rng)
    c = t.distx(a, b) + 1e-32
    ranked = sorted(rows, key=lambda r: project(t, a, b, c, r))
    h = len(ranked) // 2
    mid = project(t, a, b, c, ranked[h])
    return Node(a, b, c, mid, grow(t, ranked[:h], rng, stop), grow(t, ranked[h:], rng, stop))


def leaf(t, tree, row):
    """The leaf a row falls into."""
    while isinstance(tree, Node):
        tree = tree.left if project(t, tree.a, tree.b, tree.c, row) < tree.mid else tree.right
    return tree


def families(t, rng, sample=2000, stop=250):
    """Group every row of the table: grow a tree on a sample, then drop all rows down it."""
    tree = grow(t, rng.sample(t.rows, min(sample, len(t.rows))), rng, stop)
    groups = {}
    for r in t.rows:
        groups.setdefault(leaf(t, tree, r), []).append(r)
    return list(groups.values())
