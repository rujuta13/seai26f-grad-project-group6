import numpy as np

from step1 import _load, best
from step2 import _columns, guess
from step3 import same
from step4 import _goal_layout, _goal_vec


def _standardize(train_csv):
    G = _load(train_csv)["G"]
    mu = G.mean(axis=0)
    sd = G.std(axis=0)
    sd[sd == 0] = 1.0
    return G, mu, sd, (G - mu) / sd


def _goal_names(train_csv):
    maxg, ming = _goal_layout(train_csv)
    return maxg + ming


def _is_num(v):
    try:
        return v is not None and bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def explain(train_csv) -> str:
    inputs, _ = _columns(train_csv)
    G, mu, sd, Z = _standardize(train_csv)
    n = len(G)
    if n == 0 or G.shape[1] == 0 or not inputs:
        return "Not enough data (need rows, flags and goals) to explain anything."

    d = _load(train_csv)
    rows, X = d["rows"], d["X"]
    score = Z.mean(axis=1)
    names = _goal_names(train_csv)

    lines = [
        f"Data: {n} measured builds, {len(inputs)} on/off flags, "
        f"goals: {', '.join(names)}.",
        "Score: each goal is z-scored against the training data (minimised goals "
        "negated, so higher is better); a build's score is the mean of its z-scores.",
    ]

    # Recommended flag set: the best-scoring Pareto-front build.
    front = best(train_csv)
    if front:
        gi = max(front, key=lambda i: score[i])
        rec = rows[gi]
        on_flags = [c for j, c in enumerate(inputs)
                    if X[gi, j] != X[:, j].min()]
        lines.append(
            f"Recommended build: training row {gi}, score {score[gi]:.2f}; "
            "no other build beats it on every goal."
        )
        lines.append(
            "Flags on: " + (", ".join(on_flags) if on_flags else "none") + "."
        )
        lines.append(
            "Measured goals: " + ", ".join(
                f"{g} = {rec[g]:.6g}" for g in names if _is_num(rec.get(g))
            ) + "."
        )
        est = guess(train_csv, rec)
        if est:
            lines.append(
                "Model estimate for this flag set (random forest, 10th-90th "
                "percentile of held-out errors; fitted on the same rows, so "
                "optimistic): " + "; ".join(
                    f"{g} {mid:.4g} [{lo:.4g}, {hi:.4g}]"
                    for g, (mid, lo, hi) in est.items()
                ) + "."
            )

    # Which flags go with better builds, with a permutation test per flag.
    effects = []
    for j, c in enumerate(inputs):
        on = X[:, j] != X[:, j].min()
        off = ~on
        if not (on.any() and off.any()):
            continue
        diff = float(score[on].mean() - score[off].mean())
        real = not same(score[on], score[off])
        if real:
            effects.append((abs(diff), c, diff, float(score[on].mean()),
                            float(score[off].mean())))
    effects.sort(key=lambda t: -t[0])

    if effects:
        lines.append(
            "Flags whose on/off split changes the score beyond run-to-run noise "
            f"(permutation test, 5%; {len(inputs)} flags tested without "
            "correction, so treat marginal ones as hypotheses):"
        )
        for _, c, diff, m_on, m_off in effects[:8]:
            verb = "helps" if diff > 0 else "hurts"
            lines.append(
                f"  {c} {verb}: mean score {m_on:.2f} when on vs {m_off:.2f} when off."
            )
    else:
        lines.append(
            "No single flag's on/off split is distinguishable from noise in this data."
        )

    return "\n".join(lines)


def why(train_csv, row) -> str:
    G, mu, sd, Z = _standardize(train_csv)
    if len(G) == 0 or G.shape[1] == 0:
        return "Cannot judge this build: the training data has no goal columns."

    maxg, ming = _goal_layout(train_csv)
    names = maxg + ming
    median = float(np.median(Z.mean(axis=1)))

    measured = all(_is_num(row.get(g)) for g in names)
    if measured:
        vec = np.array(_goal_vec(row, maxg, ming), dtype=float)
        source = "measured"
    else:
        est = guess(train_csv, row)
        mid_row = {g: est[g][0] for g in names if g in est}
        vec = np.array(_goal_vec(mid_row, maxg, ming), dtype=float)
        source = "estimated"

    z = (vec - mu) / sd
    s = float(z.mean())
    verdict = "good" if s >= median else "bad"
    k = int(np.sum(np.all(Z >= z, axis=1) & np.any(Z > z, axis=1)))
    dom_text = ("no training build beats it on every goal" if k == 0
                else f"{k} training build(s) beat it on every goal")
    first = (f"Rated {verdict}: {source} goals give an average standardized score of "
             f"{s:.2f} against a training median of {median:.2f}, and {dom_text}.")

    i_min, i_max = int(np.argmin(z)), int(np.argmax(z))
    second = (f"Weakest goal: {names[i_min]} (z {z[i_min]:.2f}); strongest: "
              f"{names[i_max]} (z {z[i_max]:.2f})")
    if not measured:
        second += "; goal values are model estimates, not measurements"
    return first + " " + second + "."

