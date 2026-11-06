# The Shrinking-Code Experiment, at Scale

CSC591 SE for AI, NC State, Fall '26: graduate project, group 6.

Does a well-factored library make each new AI skill cheaper to write? We climb a ladder of
ten skills (remember, guess, certify, flow, blame, justify, choose, fix, spot, race) across
three domains, seven ways, and count the new code per step.

| Steps | Domain | Client | MOOT data |
|---|---|---|---|
| step1–step10 | SS-W | compiler performance engineer | `config/SS-W.csv` |
| step11–step20 | players | football club scout | `behavior_data/all_players.csv` |
| step21–step30 | marketing | retail marketing manager | `sales_data/Marketing_Analytics.csv` |

- Claim (pre-registered): [docs/claim.md](docs/claim.md)
- Requirements: [docs/requirements.md](docs/requirements.md)

## Run

```sh
make data     # download the MOOT csvs, unmodified
make test     # all tests
make loc      # new lines of code per step tag
make step1    # each step's demo (added as steps land)
```

## Layout

```
src/        shared library (treatment 7: written by us)
domains/    per-domain config and demos
harness/    LLM treatments 1-6, LOC counting
tests/      one test file per step
data/       download script; csvs are not committed
docs/       requirements, claim
```

Step tags: `step0` is the empty scaffold, so `step1`'s new code starts from zero.
