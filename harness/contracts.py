"""Step contracts: what each step's code must provide, and how the harness scores it.

Every treatment (T1-T7) builds the same ten skills per domain. For step N the code lives in
`stepN.py` (earlier steps' files sit beside it and may be imported). That file must define
the functions in the step's API below. The harness imports it and calls those functions on
the shared split (harness/data.py); the code never sees the test file except where an API
hands it rows.

Conventions given to every writer (LLM or human):
- Data files are MOOT csvs: column names ending in "+" are goals to maximise, "-" goals to
  minimise; names ending in "X" are ignored; everything else is an input.
- A *row* is a dict {column name: value}; numbers are floats, other cells strings.
- Row indices count data rows of a csv from 0 (header excluded).

Each scorer returns {"ok": bool, "score": float|None, "baseline": float|None,
"better": "lower"|"higher"|None, "extra": {...}}.
"""
import csv
import math
import os
import random
import re
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIREMENTS = os.path.join(ROOT, "docs", "requirements.md")
DOMAIN_ORDER = ["ssw", "players", "marketing"]        # order of domains in requirements.md
SKILLS = ["remember", "guess", "certify", "flow", "blame",
          "justify", "choose", "fix", "spot", "race"]
SEED = 20261120


# ---------------------------------------------------------------- the asks, from the doc

def asks(domain):
    """{skill: [ask, ask]} for one domain, read from docs/requirements.md."""
    out, d = {}, -1
    with open(REQUIREMENTS) as f:
        for line in f:
            if line.startswith("## Domain"):
                d += 1
            m = re.match(r"\|\s*\d+\s*\|\s*(.+?)\s*\|\s*(\w+)\s*\|\s*$", line)
            if m and d >= 0 and DOMAIN_ORDER[d] == domain and m.group(2) in SKILLS:
                out.setdefault(m.group(2), []).append(m.group(1))
    return out


def step_of(domain, n):
    """Global step number -> skill name (steps 1-10 ssw, 11-20 players, 21-30 marketing)."""
    assert DOMAIN_ORDER[(n - 1) // 10] == domain, f"step {n} is not in domain {domain}"
    return SKILLS[(n - 1) % 10]


# ---------------------------------------------------------------- data helpers (harness side)

def cell(v):
    try:
        return float(v)
    except ValueError:
        return v.strip()


def read(path):
    """-> (header, rows as dicts)."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        return header, [dict(zip(header, map(cell, row))) for row in r]


def goals(header):
    return [h for h in header if h[-1] in "+-"]


def inputs(header):
    return [h for h in header if h[-1] not in "+-X"]


class Heaven:
    """Distance to heaven: 0 = best on every goal, 1 = worst. Normalised over `rows`."""

    def __init__(self, header, rows):
        self.goals = goals(header)
        self.lo = {g: min(r[g] for r in rows) for g in self.goals}
        self.hi = {g: max(r[g] for r in rows) for g in self.goals}

    def __call__(self, row, only=None):
        gs = [only] if only else self.goals
        d = 0
        for g in gs:
            x = (row[g] - self.lo[g]) / (self.hi[g] - self.lo[g] + 1e-32)
            x = min(1, max(0, x))
            d += (x - (1 if g.endswith("+") else 0)) ** 2
        return math.sqrt(d / len(gs))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def result(ok, score=None, baseline=None, better=None, **extra):
    return {"ok": bool(ok), "score": score, "baseline": baseline, "better": better, "extra": extra}


def blank_goals(header, rows, path):
    """Write `rows` with every goal cell set to '?', for skills that must not see goals."""
    gs = set(goals(header))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["?" if h in gs else _fmt(r[h]) for h in header])


def _fmt(v):
    return repr(v) if isinstance(v, float) and not v.is_integer() else (
        str(int(v)) if isinstance(v, float) else v)


def without(row, cols):
    return {k: v for k, v in row.items() if k not in cols}


# ---------------------------------------------------------------- the ten contracts
# Each entry: the API text shown to the writer, and the scorer.
# `env` carries: header, train, test (rows), full (every row of the MOOT file),
# train_csv, test_csv, blind_csv (train with goals '?'), heaven, tmp (scratch dir).

API = {
"remember": """\
def families(train_csv) -> list[int]
    One family (cluster) label per data row of train_csv, in file order.
def best(train_csv) -> list[int]
    Row indices of the unusually good rows (best on all goals together), at most 5% of rows.""",

"guess": """\
def guess(train_csv, row) -> dict[str, tuple[float, float, float]]
    `row` has its goal columns removed. Return {goal: (estimate, low, high)} for every goal,
    where low..high is the range you are fairly sure the true value falls in.""",

"certify": """\
def same(xs, ys) -> bool
    xs, ys: lists of numbers (e.g. run times). True if they are NOT really different
    (the gap is noise), False if the difference is real.""",

"flow": """\
def stream(train_csv, rows) -> list[bool]
    `rows` arrive one at a time, in order. For each, decide keep (True) or not, using only
    train_csv and the rows before it. Keep the ones worth keeping (good on the goals).
def drift(train_csv, rows) -> int | None
    Index into `rows` of the first row at which you warn that new rows have stopped looking
    like train_csv, or None if they never do.""",

"blame": """\
def blame(train_csv) -> list[str]
    Input column names, most to blame for bad rows (bad on the goals) first.""",

"justify": """\
def explain(train_csv) -> str
    Reasons, in plain words a skeptical expert can check, for what makes rows good.
def why(train_csv, row) -> str
    One or two sentences on why this row (goals included) is good or bad.""",

"choose": """\
def choose(blind_csv, label, budget=30, goal=None) -> int
    blind_csv has every goal cell set to '?'. label(i) returns the goals of row i as a dict;
    it may be called at most `budget` times (more raises an error). Return the index of the
    best row found. If `goal` names one goal, optimise only that goal.""",

"fix": """\
def fix(train_csv, row, frozen=()) -> dict
    `row` is a bad row (goals removed). Return the same row's inputs with as few values
    changed as possible so it becomes good. Columns named in `frozen` must not change.""",

"spot": """\
def odd(train_csv, rows) -> list[float]
    One suspicion score per row in `rows` (goals included): higher means more likely a
    glitch, a typo, or unlike anything in train_csv.""",

"race": """\
def race(train_csv, test_csv) -> dict[str, float]
    Compare your earlier skills against standard rivals (e.g. random search, an
    off-the-shelf model) on test_csv. Return {method: score}, including the key "ours".""",
}


def score_remember(m, env):
    fam = list(m.families(env["train_csv"]))
    best = list(m.best(env["train_csv"]))
    n, h = len(env["train"]), env["heaven"]
    ok = len(fam) == n and 2 <= len(set(fam)) <= n // 2 and 1 <= len(best) <= max(1, n // 20)
    d = [h(r) for r in env["train"]]
    by = {}
    for f, x in zip(fam, d):
        by.setdefault(f, []).append(x)
    spread = mean(statistics.pstdev(v) for v in by.values()) / (statistics.pstdev(d) + 1e-32)
    return result(ok, mean(d[i] for i in best if 0 <= i < n), mean(d), "lower",
                  families=len(set(fam)), within_family_spread=spread)


def score_guess(m, env):
    gs, train, test = goals(env["header"]), env["train"], env["test"]
    err = {g: [] for g in gs}
    base = {g: [] for g in gs}
    hit = 0
    for r in test:
        out = m.guess(env["train_csv"], without(r, gs))
        for g in gs:
            est, lo, hi = out[g]
            span = env["heaven"].hi[g] - env["heaven"].lo[g] + 1e-32
            err[g].append(abs(est - r[g]) / span)
            base[g].append(abs(mean(t[g] for t in train) - r[g]) / span)
            hit += lo <= r[g] <= hi
    return result(True, mean(mean(v) for v in err.values()), mean(mean(v) for v in base.values()),
                  "lower", coverage=hit / (len(test) * len(gs)))


def score_certify(m, env):
    rng, wrong, n = random.Random(SEED), 0, 40
    for i in range(n):
        shift = 0 if i % 2 == 0 else 1.0
        xs = [rng.gauss(10, 1) for _ in range(30)]
        ys = [rng.gauss(10 + shift, 1) for _ in range(30)]
        wrong += bool(m.same(xs, ys)) != (shift == 0)
    return result(True, wrong / n, 0.5, "lower")


def score_flow(m, env):
    h, test = env["heaven"], env["test"]
    keep = list(m.stream(env["train_csv"], test))
    kept = [h(r) for r, k in zip(test, keep) if k]
    ok = len(keep) == len(test) and 0 < len(kept) < len(test)
    # drift: 100 ordinary rows, then 100 of the worst rows in the whole file
    rng = random.Random(SEED)
    worst = sorted(env["full"], key=h)[-len(env["full"]) // 4:]
    rows = rng.sample(test, 100) + rng.sample(worst, 100)
    at = m.drift(env["train_csv"], rows)
    return result(ok, mean(kept), mean(h(r) for r in test), "lower",
                  kept=len(kept), drift_at=at, drift_hit=at is not None and 100 <= at < 200,
                  false_alarm=at is not None and at < 100)


def effect(rows, col, h):
    """How much the mean distance-to-heaven moves across the values of one input."""
    by = {}
    for r in rows:
        by.setdefault(r[col], []).append(h(r))
    ms = [mean(v) for v in by.values() if len(v) >= 5]
    return max(ms) - min(ms) if len(ms) > 1 else 0


def score_blame(m, env):
    cols, h, test = inputs(env["header"]), env["heaven"], env["test"]
    named = [c for c in m.blame(env["train_csv"]) if c in cols][:3]
    effects = {c: effect(test, c, h) for c in cols}
    ok = len(named) > 0
    return result(ok, mean(effects[c] for c in named) if ok else None, mean(effects.values()),
                  "higher", named=named)


def score_justify(m, env):
    cols = inputs(env["header"]) + goals(env["header"])
    a = m.explain(env["train_csv"])
    b = m.why(env["train_csv"], env["test"][0])
    ok = all(isinstance(s, str) and s.strip() for s in (a, b))
    mentions = sum(c in a for c in cols)
    return result(ok and mentions > 0, None, None, None, columns_mentioned=mentions,
                  explain_words=len(a.split()), why_words=len(b.split()))


class Budget(Exception):
    pass


def score_choose(m, env, budget=30, repeats=5):
    h, train, gs = env["heaven"], env["train"], goals(env["header"])
    picks = []
    for _ in range(repeats):
        calls = [0]

        def label(i):
            calls[0] += 1
            if calls[0] > budget:
                raise Budget(f"more than {budget} labels")
            return {g: train[i][g] for g in gs}

        picks.append(h(train[m.choose(env["blind_csv"], label, budget)]))
    rng = random.Random(SEED)
    rand = [min(h(r) for r in rng.sample(train, budget)) for _ in range(20)]
    return result(True, mean(picks), mean(rand), "lower", best_possible=min(h(r) for r in train))


def score_fix(m, env, n=10):
    h, header = env["heaven"], env["header"]
    cols, gs = inputs(header), goals(header)
    index = {tuple(r[c] for c in cols): r for r in env["full"]}

    def nearest(x):
        key = tuple(x.get(c) for c in cols)
        if key in index:
            return index[key]
        return min(env["full"], key=lambda r: sum(r[c] != x.get(c) for c in cols))

    bad = sorted(env["test"], key=h)[-n:]
    before, after, flips, frozen_ok = [], [], [], True
    for i, r in enumerate(bad):
        frozen = (cols[0],) if i % 2 else ()
        new = m.fix(env["train_csv"], without(r, gs), frozen)
        frozen_ok &= all(new.get(c) == r[c] for c in frozen)
        flips.append(sum(new.get(c) != r[c] for c in cols))
        before.append(h(r))
        after.append(h(nearest(new)))
    return result(frozen_ok, mean(after), mean(before), "lower", mean_changes=mean(flips),
                  frozen_respected=frozen_ok)


def score_spot(m, env, n_bad=10):
    rng, header = random.Random(SEED), env["header"]
    gs = goals(header)
    rows = [dict(r) for r in rng.sample(env["test"], len(env["test"]) - n_bad)]
    bad = []
    for r in rng.sample(env["test"], n_bad):
        r = dict(r)
        g = rng.choice(gs)
        r[g] = r[g] * 10 if r[g] else 999999.0       # a glitch: one goal off by 10x
        bad.append(r)
    allrows = rows + bad
    order = list(range(len(allrows)))
    rng.shuffle(order)
    shown = [allrows[i] for i in order]
    s = list(m.odd(env["train_csv"], shown))
    top = sorted(range(len(shown)), key=lambda i: -s[i])[:n_bad]
    caught = sum(order[i] >= len(rows) for i in top)
    return result(len(s) == len(shown), caught / n_bad, n_bad / len(shown), "higher")


def score_race(m, env):
    out = m.race(env["train_csv"], env["test_csv"])
    ok = isinstance(out, dict) and "ours" in out and len(out) >= 2 and all(
        isinstance(v, (int, float)) for v in out.values())
    return result(ok, None, None, None, **({k: float(v) for k, v in out.items()} if ok else {}))


SCORE = {s: globals()[f"score_{s}"] for s in SKILLS}


def brief(domain, n):
    """The contract text for step n: the client's two asks plus the API to provide."""
    skill = step_of(domain, n)
    a = asks(domain)[skill]
    lines = [f"Step {n}: {skill}.", "The client asks:"] + [f'  - "{x}"' for x in a]
    lines += ["", f"Write step{n}.py defining exactly these functions:", API[skill]]
    return "\n".join(lines)
