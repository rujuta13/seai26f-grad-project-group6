# Pre-registered claim

**Status: final, 2026-11-15, committed at tag `step0`, before any skill code existed. This
text is never edited. Any later change goes in the dated "Amendments" section below, with
the reason.**

## The two hypotheses (verbatim from the project spec)

> **The Timm hypothesis.** Neuro-symbolic systems improve when their symbolic half is
> factored into small reusable structures. The ten skills are synonyms: one machine
> (cluster, tree, leaf) wearing ten hats. Hence an LLM, or a human, extends a well-factored
> library with less new code, and fewer errors, than a monolithic one.

> **The rival.** The ten skills are ten separate problems. Factoring buys nothing: skill i+1
> costs about the same whoever writes it, and whatever library sits underneath.

## Domains and data

Three clients on three MOOT data sets, one shared codebase, used exactly as MOOT ships them:
a compiler performance engineer tuning 16 LLVM passes (`config/SS-W.csv`, 65,536 rows,
2 goals); a football scout looking for strong penalty-takers (`behavior_data/all_players.csv`,
17,737 rows, 2 goals); and a retail marketing manager targeting campaigns
(`sales_data/Marketing_Analytics.csv`, 2,205 rows, 8 goals). Ten skills per domain, one
git-tagged step per skill: steps 1–10 SS-W, 11–20 players, 21–30 marketing. The first domain
builds the library, and domains 2 and 3 test whether it transfers.

## Measures

- **New code per step:** non-blank, non-comment Python lines added under `src/` and
  `domains/` between consecutive tags (`harness/loc.py`); for the LLM treatments, lines in
  the code the LLM returns for that step.
- **Correctness per step:** whether the step's tests pass, plus the skill's own score where
  it has one (e.g. prediction error for *guess*, distance-to-heaven of the pick for *choose*).
- **Comparisons:** only differences that `same` (Cliff's delta + Kolmogorov-Smirnov,
  from ezr) certifies are reported; everything else is called a tie.
- **Treatments:** the seven in the spec; N ≥ 10 fresh, memory-free trials per LLM treatment.

## Win conditions

**The Timm hypothesis wins if all three hold:**

1. **Bars fall.** In each domain, new code in its last five steps is smaller than in its
   first five, and `same` says the difference is real (treatment 7 and reuse treatments 2, 4, 6).
2. **Reuse beats no-reuse.** For each library (plain Python, scikit-learn, ezr), the
   reuse treatment writes less new code per step than its no-reuse twin (2 < 1, 4 < 3,
   6 < 5), certified by `same`, **without** being worse on correctness.
3. **Transfer.** Domains 2 and 3 each need less than half the new code that domain 1
   needed (treatment 7).

**The rival wins if:** new code per step is a `same`-certified tie between the first and
last halves of each domain, **and** each reuse treatment ties with its no-reuse twin.

**Anything else is a split verdict.** We report which conditions held, per domain and
overall, and do not round it up to a win for either side.

## Amendments

**2026-11-22: N per LLM treatment is 3–5, not ≥ 10.** The course instructor ruled that 3–5
repeats per LLM treatment is enough, to keep runtime and token use within the course
budget. We ran T1, T2, T5 and T6 five times over all 30 steps, and T3 and T4 three times
over steps 1–10. Smaller N makes `same` less likely to certify a difference, so it biases
the comparisons toward ties, not toward either hypothesis. Analysis: `harness/claims.py`.
