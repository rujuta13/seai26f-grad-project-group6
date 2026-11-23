import numpy as np

from step1 import _load
from step2 import _columns, guess
from step4 import _goal_layout, _goal_vec
from step6 import _standardize, _goal_names, _is_num

_REPS = 5000   # permutations per flag test
_SEED = 0
_ALPHA = 0.05  # family-wise level, shared across all flags tested


def _perm_p(a, b, rng):
    """Two-sided permutation p-value for the difference of means of a and b."""
    obs = abs(float(a.mean()) - float(b.mean()))
    if obs == 0.0:
        return 1.0
    pooled = np.concatenate([a, b])
    n = len(a)
    count = 0
    for _ in range(_REPS):
        p = rng.permutation(pooled)
        if abs(float(p[:n].mean()) - float(p[n:].mean())) >= obs - 1e-12:
            count += 1
    return (count + 1) / (_REPS + 1)


def _on_mask(col):
    """A flag counts as 'on' when it is above its lowest training value."""
    return col != col.min()


def explain(train_csv) -> str:
    """Plain-language reasons that separate good rows from the rest, checkable against the data."""
    inputs, _ = _columns(train_csv)
    G, mu, sd, Z = _standardize(train_csv)
    n = len(G)
    if n < 2 or G.shape[1] == 0:
        return "Not enough measured rows to give reasons."

    score = Z.mean(axis=1)
    names = _goal_names(train_csv)
    X = _load(train_csv)["X"]
    top_cut = float(np.quantile(score, 0.95))
    top = score >= top_cut

    lines = [
        f"These reasons come from {n} measured rows. Each row is scored by averaging "
        f"its goals, each goal standardized so higher is better (minimised goals reversed): "
        f"{', '.join(names)}.",
        f"The top 5% of rows (score at least {top_cut:.2f}) are the reference for 'good'.",
    ]

    # Test each on/off flag against the score, with a Bonferroni correction for
    # the number of flags tested, so that chance hits are not reported as reasons.
    rng = np.random.default_rng(_SEED)
    testable = []
    for j, c in enumerate(inputs):
        on = _on_mask(X[:, j])
        off = ~on
        if on.any() and off.any():
            testable.append((j, c, on, off))

    m = len(testable)
    reasons = []
    if m:
        cutoff = _ALPHA / m
        for j, c, on, off in testable:
            diff = float(score[on].mean() - score[off].mean())
            p = _perm_p(score[on], score[off], rng)
            if p < cutoff:
                reasons.append((abs(diff), c, diff, on, off, p))
    reasons.sort(key=lambda t: -t[0])

    if reasons:
        lines.append(
            f"Flags whose on/off difference holds up after correcting for {m} flags "
            f"tested (family-wise 5%):"
        )
        for _, c, diff, on, off, p in reasons[:6]:
            share_top = float(np.mean(on[top])) if top.any() else float("nan")
            share_all = float(np.mean(on))
            verb = "goes with better rows" if diff > 0 else "goes with weaker rows"
            lines.append(
                f"  - {c} being on {verb}: average score {score[on].mean():.2f} when on "
                f"vs {score[off].mean():.2f} when off; it is on in {share_top:.0%} of "
                f"top-5% rows against {share_all:.0%} of all rows."
            )
        lines.append(
            "These are associations in past data, not proof that changing the flag "
            "causes the difference. Check them against the recorded rows before relying on them."
        )
    else:
        lines.append(
            f"No single flag's on/off difference survives correction for {m} flags tested, "
            "so this data gives no flag-level reason you can check."
        )

    return "\n".join(lines)


def why(train_csv, row) -> str:
    """One or two sentences on why this row, goals included, is good or bad."""
    G, mu, sd, Z = _standardize(train_csv)
    if len(G) == 0 or G.shape[1] == 0:
        return "Cannot judge this row: the training data has no goal columns."

    maxg, ming = _goal_layout(train_csv)
    names = maxg + ming
    median = float(np.median(Z.mean(axis=1)))

    measured = all(_is_num(row.get(g)) for g in names)
    if measured:
        vec = np.array(_goal_vec(row, maxg, ming), dtype=float)
        raw = {g: float(row[g]) for g in names}
        source = "measured"
    else:
        est = guess(train_csv, row)
        if not all(g in est for g in names):
            return "Cannot judge this row: the goal estimates are incomplete."
        mids = {g: est[g][0] for g in names}
        vec = np.array(_goal_vec(mids, maxg, ming), dtype=float)
        raw = mids
        source = "estimated"

    z = (vec - mu) / sd
    s = float(z.mean())
    verdict = "good" if s >= median else "weak"
    i_min, i_max = int(np.argmin(z)), int(np.argmax(z))

    first = (
        f"This row rates {verdict}: its {source} goals average a standardized score of "
        f"{s:.2f} against a training median of {median:.2f}."
    )
    second = (
        f"Weakest on {names[i_min]} ({raw[names[i_min]]:.4g}, z {z[i_min]:.2f}); "
        f"strongest on {names[i_max]} ({raw[names[i_max]]:.4g}, z {z[i_max]:.2f})"
    )
    if not measured:
        second += "; these goal values are model estimates, not measurements, so check them on video"
    return first + " " + second + "."

