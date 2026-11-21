"""step6.py - client question 7: justify.

explain(train_csv) gives plain-words reasons, a skeptical senior engineer can
check, for what makes builds good. why(train_csv, row) says in one or two
sentences why a single build is rated good or bad.

Reused from earlier steps:
  step1: _load (MOOT csv loading), _dominates (goal dominance), best (the
         non-dominated builds, used as the "good" group)
  step5: BAD_FRACTION (share of builds called bad), _percentiles (goal
         percentiles), _effect (standardised bad-minus-good difference)
"""
import bisect
import functools
from types import SimpleNamespace

from step1 import _dominates, _load, best
from step5 import BAD_FRACTION, _effect, _percentiles


def _pct_of(value, sorted_vals):
    """Percentile of a new value on the same scale as step5._percentiles:
    the mean rank of its ties among the training values, divided by n - 1."""
    n = len(sorted_vals)
    if n == 1:
        return 1.0
    lo = bisect.bisect_left(sorted_vals, value)
    hi = bisect.bisect_right(sorted_vals, value)
    if lo < hi:
        rank = (lo + hi - 1) / 2
    else:
        rank = min(max(lo - 0.5, 0.0), n - 1.0)
    return rank / (n - 1)


@functools.lru_cache(maxsize=None)
def _model(train_csv):
    """Everything explain and why need, computed once per training file."""
    input_cols, goal_cols, X, Y = _load(train_csv)
    n, d, m = len(X), len(input_cols), len(goal_cols)
    if n == 0:
        raise ValueError("training file has no data rows")
    if m == 0:
        raise ValueError("training file has no goal columns")

    # score = mean goal percentile (1 = best), as in step5.blame's badness
    goal_sorted = [sorted(Y[i][g] for i in range(n)) for g in range(m)]
    pct_cols = [_percentiles([Y[i][g] for i in range(n)]) for g in range(m)]
    score = [sum(col[i] for col in pct_cols) / m for i in range(n)]

    # bad = worst BAD_FRACTION by score, same tie-break as step5.blame
    k = max(1, int(BAD_FRACTION * n))
    order = sorted(range(n), key=lambda i: (score[i], i))
    cut = score[order[k - 1]]
    bad_idx = sorted(order[:k])
    bad_set = set(bad_idx)

    # good = non-dominated builds (step1.best); fall back to the rest
    good_idx = best(train_csv)
    if not good_idx:
        good_idx = [i for i in range(n) if i not in bad_set] or list(range(n))

    eff, bad_mean, good_mean = [], [], []
    for t in range(d):
        bv = [X[i][t] for i in bad_idx]
        gv = [X[i][t] for i in good_idx]
        bad_mean.append(sum(bv) / len(bv))
        good_mean.append(sum(gv) / len(gv))
        eff.append(_effect(bv, gv))
    ranked = sorted(range(d), key=lambda t: (-eff[t], t))

    return SimpleNamespace(
        input_cols=input_cols, goal_cols=goal_cols, X=X, Y=Y,
        goal_sorted=goal_sorted, score=score, cut=cut,
        good_idx=good_idx, eff=eff, bad_mean=bad_mean,
        good_mean=good_mean, ranked=ranked,
    )


def explain(train_csv) -> str:
    """Plain-words reasons for what makes builds good, checkable by a skeptic."""
    M = _model(train_csv)
    n = len(M.X)
    d = len(M.input_cols)
    names = [name for name, _ in M.goal_cols]

    # recommended: the best-scoring non-dominated build
    rec = min(M.good_idx, key=lambda i: (-M.score[i], i))
    goals = ", ".join(f"{name} = {sign * M.Y[rec][g]:g}"
                      for g, (name, sign) in enumerate(M.goal_cols))
    on = [f"{c}={M.X[rec][t]:g}"
          for t, c in enumerate(M.input_cols) if M.X[rec][t] != 0]

    up = [t for t in M.ranked if M.eff[t] > 0][:5]
    down = [t for t in sorted(range(d), key=lambda t: (M.eff[t], t))
            if M.eff[t] < 0][:5]

    def line(t):
        return (f"  {M.input_cols[t]}: bad-build mean {M.bad_mean[t]:.2f}, "
                f"good-build mean {M.good_mean[t]:.2f}, effect {M.eff[t]:+.2f}")

    lines = [
        f"Basis: {n} training builds, {d} input settings, goals: {', '.join(names)}.",
        ("How a build is judged: each goal is turned into a percentile among the "
         "training builds, higher is better (minimised goals are inverted), and a "
         "build's score is its mean percentile. "
         f"The worst {100 * BAD_FRACTION:.0f}% by score are called bad."),
        ("A good build is non-dominated: no other training build is at least as good "
         "on every goal and strictly better on at least one."),
        "",
        "Recommended set: the non-dominated build with the highest score.",
        f"  score: {100 * M.score[rec]:.0f} of 100",
        f"  goals (original units): {goals}",
        f"  settings on ({len(on)} of {d}): {', '.join(on) if on else 'none'}",
        "",
        ("What separates bad builds from good ones: standardised difference, "
         "bad minus good (positive = more common in bad builds)."),
        "More common in bad builds:",
    ]
    lines += [line(t) for t in up] or ["  none"]
    lines.append("More common in good builds:")
    lines += [line(t) for t in down] or ["  none"]
    lines += [
        "",
        ("Caveats: these are associations found in this one training set, not proof "
         "that a setting causes the change in any goal. Settings interact, so one "
         "setting's effect depends on the others, and percentiles hide how large the "
         "gaps are in original units. Before relying on the recommendation, re-run it "
         "on fresh builds and use step3.same to check that goal differences exceed "
         "run-to-run noise."),
    ]
    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences on why this build (goals included) is good or bad."""
    M = _model(train_csv)
    names = [name for name, _ in M.goal_cols]
    vec = [sign * float(row[name]) for name, sign in M.goal_cols]
    pcts = [_pct_of(v, M.goal_sorted[g]) for g, v in enumerate(vec)]
    score = sum(pcts) / len(pcts)
    verdict = "bad" if score <= M.cut else "good"
    lo = min(range(len(pcts)), key=lambda g: pcts[g])
    hi = max(range(len(pcts)), key=lambda g: pcts[g])

    s1 = (f"Rated {verdict}: its mean goal percentile is {100 * score:.0f} "
          f"(0 = worst, 100 = best), against a cut-off of {100 * M.cut:.0f} "
          f"for the worst {100 * BAD_FRACTION:.0f}% of training builds; "
          f"{names[lo]} is weakest at {100 * pcts[lo]:.0f} and "
          f"{names[hi]} strongest at {100 * pcts[hi]:.0f}.")

    dominated = any(_dominates(y, vec) for y in M.Y)
    if dominated:
        s2 = ("Some training build is at least as good on every goal and better on "
              "one, so it is not on the non-dominated front.")
    else:
        s2 = ("No training build beats it on every goal together, so no training "
              "build dominates it.")

    if verdict == "bad":
        sus = []
        for t in M.ranked:
            if M.eff[t] <= 0 or len(sus) == 3:
                break
            c = M.input_cols[t]
            if c in row and float(row[c]) > M.good_mean[t]:
                sus.append(c)
        if sus:
            s2 += (" Flags most tied to bad builds that it sets above the usual "
                   "good-build level: " + ", ".join(sus) + ".")

    return f"{s1} {s2}"

