"""Test win conditions 1 and 2 of docs/claim.md on results/results.csv, using ezr's `same`.

    python3 harness/claims.py

`same(xs, ys)` is True when Cliff's delta AND Kolmogorov-Smirnov both call two samples
indistinguishable; a difference counts as certified only when `same` is False.
Repeat 0 rows (smoke tests) are left out.

Condition 1, bars fall: per domain and treatment, new_loc of the domain's first five
  steps vs its last five, pooled over repeats. Holds if last < first and certified.
Condition 2, reuse beats no-reuse: per library, new_loc per step of the reuse twin vs the
  no-reuse twin, per domain and overall. Holds if reuse < no-reuse, certified, and
  reuse is not worse on correctness: neither its ran_ok rate nor its beats-baseline rate
  per step is lower by a certified difference.
"""
import csv
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "libs"))
import ezr  # noqa: E402

RESULTS = os.path.join(os.path.dirname(HERE), "results", "results.csv")
DOMAINS = {"ssw": range(1, 11), "players": range(11, 21), "marketing": range(21, 31)}
TWINS = {"python": ("1", "2"), "sklearn": ("3", "4"), "ezr": ("5", "6")}


def rows():
    with open(RESULTS, newline="") as f:
        return [r for r in csv.DictReader(f) if r["repeat"] != "0"]


def beats(r):
    if not r["score"]:
        return None
    s, b = float(r["score"]), float(r["baseline"])
    return s < b if r["better"] == "lower" else s > b


def verdict(xs, ys):
    """'less' / 'more' if xs differ from ys (certified), else 'tie'."""
    if len(xs) < 2 or len(ys) < 2:
        return "n/a"
    if ezr.same(xs, ys):
        return "tie"
    return "less" if st.mean(xs) < st.mean(ys) else "more"


def loc(rs, t, steps):
    return [int(r["new_loc"]) for r in rs if r["treatment"] == t and int(r["step"]) in steps]


def per_step(rs, t, steps, f):
    """One 0/1 outcome per step row, like new_loc (claim: "correctness per step"); None skipped.
    Per-trial rates would leave only 3-5 heavily tied values, where one failure in 50
    steps already pushes Cliff's delta past 0.197."""
    out = []
    for r in rs:
        if r["treatment"] == t and int(r["step"]) in steps:
            v = f(r)
            if v is not None:
                out.append(float(v))
    return out


def reps(rs, t, steps):
    return len({r["repeat"] for r in rs if r["treatment"] == t and int(r["step"]) in steps})


def condition1(rs):
    print("## Condition 1: bars fall (new code, first five steps vs last five, per domain)\n")
    print(f"{'treatment':10} {'domain':10} {'reps':>4} {'first5 mean':>11} {'last5 mean':>10}  verdict")
    held = {}
    for t in ("2", "4", "6"):
        for d, steps in DOMAINS.items():
            first, last = loc(rs, t, steps[:5]), loc(rs, t, steps[5:])
            n = reps(rs, t, steps)
            if not first:
                print(f"T{t:9} {d:10} {0:>4} {'':>11} {'':>10}  no data")
                held[(t, d)] = None
                continue
            v = verdict(last, first)
            held[(t, d)] = v == "less"
            print(f"T{t:9} {d:10} {n:>4} {st.mean(first):11.1f} {st.mean(last):10.1f}  "
                  f"{'HOLDS (falls)' if v == 'less' else 'rises' if v == 'more' else v}")
    return held


def condition2(rs):
    print("\n## Condition 2: reuse beats no-reuse (new code per step, and correctness)\n")
    print(f"{'library':8} {'domain':10} {'reps':>9} {'loc no-reuse':>12} {'loc reuse':>9} "
          f"{'loc':>5} {'ran_ok':>13} {'ok-verdict':>10} {'beats-base':>13} {'bb-verdict':>10}  holds?")
    held = {}
    scopes = dict(DOMAINS, overall=range(1, 31))
    for lib, (a, b) in TWINS.items():
        for d, steps in scopes.items():
            la, lb = loc(rs, a, steps), loc(rs, b, steps)
            if not la or not lb:
                print(f"{lib:8} {d:10} no data")
                held[(lib, d)] = None
                continue
            v = verdict(lb, la)
            oka = per_step(rs, a, steps, lambda r: r["ran_ok"] == "True")
            okb = per_step(rs, b, steps, lambda r: r["ran_ok"] == "True")
            bba, bbb = per_step(rs, a, steps, beats), per_step(rs, b, steps, beats)
            vok, vbb = verdict(okb, oka), verdict(bbb, bba)
            ok = v == "less" and vok != "less" and vbb != "less"
            held[(lib, d)] = ok
            print(f"{lib:8} {d:10} {reps(rs, a, steps)}v{reps(rs, b, steps):<7} "
                  f"{st.mean(la):12.1f} {st.mean(lb):9.1f} {v:>5} "
                  f"{st.mean(oka):.2f}v{st.mean(okb):.2f}{'':>4} {vok:>10} "
                  f"{st.mean(bba):.2f}v{st.mean(bbb):.2f}{'':>4} {vbb:>10}  {'YES' if ok else 'no'}")
    return held


def main():
    rs = rows()
    c1 = condition1(rs)
    c2 = condition2(rs)
    print("\nverdict columns: 'less' = reuse/last-five certified lower than its comparison; "
          "'tie' = `same` cannot tell them apart; ran_ok and beats-base are per-step pass rates.")
    for name, h in (("Condition 1", c1), ("Condition 2", c2)):
        known = [v for v in h.values() if v is not None]
        missing = [k for k, v in h.items() if v is None]
        print(f"{name}: holds in {sum(known)}/{len(known)} cells with data"
              + (f"; no data for {missing}" if missing else ""))


if __name__ == "__main__":
    main()
