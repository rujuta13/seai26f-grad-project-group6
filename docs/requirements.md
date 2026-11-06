# Requirements: Three Domains

CSC591 SE for AI, Fall '26 — Graduate Project (Shrinking-Code Experiment, at Scale)

Three clients, three MOOT data sets, one shared codebase. Every ask is written in the
client's words and tagged with the skill it demos:

| Tag | Skill |
|---|---|
| remember | representation: summarise and cluster what we have |
| guess | prediction: estimate goals before measuring them |
| certify | certification: is a difference real or noise? |
| flow | streaming: judge items as they arrive one at a time |
| blame | diagnosis: what makes the bad ones bad? |
| justify | explanation: defend a verdict to a skeptic |
| choose | optimisation: find the best while measuring very few |
| fix | repair: the smallest change that turns bad into good |
| spot | anomaly detection: flag rows that look wrong |
| race | baselining: compare our skills against standard tools |

All three data sets are used exactly as MOOT ships them (no rows or columns edited).

---

## Domain 1: Compiler performance engineer

**Data:** `optimize/config/SS-W.csv` — 65,536 builds. Inputs: 16 on/off LLVM optimisation
passes (`inline`, `licm`, `gvn`, `instcombine`, ...). Goals: `Performance-` (lower is
better), `Cpu-` (lower is better).

**Client:** the build team lead for a product compiled with LLVM. Every benchmark run costs
machine time, and they want fast binaries without burning CPU in the process.

| # | Ask (client's words) | Skill |
|---|---|---|
| 1 | Out of 65,000 flag combinations, show me the handful of families they fall into. | remember |
| 2 | Which flag sets are unusually fast *and* light on CPU? | remember |
| 3 | Before I benchmark a flag set, guess how fast it will be. | guess |
| 4 | Guess the CPU cost too, and tell me how sure you are. | guess |
| 5 | Turning on `inline` looked faster last week; is that real or just run-to-run noise? | certify |
| 6 | Are my two favourite flag sets actually different, or should I pick the simpler one? | certify |
| 7 | As nightly benchmark results come in, tell me right away which ones are worth keeping. | flow |
| 8 | Warn me as soon as tonight's results stop looking like last month's. | flow |
| 9 | Why are the slow builds slow? Which passes are to blame? | blame |
| 10 | Why do some fast builds still eat so much CPU? | blame |
| 11 | Explain your recommended flag set to a senior engineer who distrusts magic. | justify |
| 12 | For any single build, tell me in one sentence why you rated it good or bad. | justify |
| 13 | Find me the best flag set while benchmarking no more than 30 builds. | choose |
| 14 | If I only care about speed (or only CPU), does the best choice change? | choose |
| 15 | This build is slow: what is the fewest flags I can flip to make it fast? | fix |
| 16 | We can't touch `inline` for legal reasons; what's the best fix without it? | fix |
| 17 | Flag any benchmark result that looks like a glitch, not a real measurement. | spot |
| 18 | Which flag sets behave unlike anything else we've seen? | spot |
| 19 | Race our tuner against a random search and a standard library's tuner. | race |
| 20 | Does our guesser beat an off-the-shelf model at predicting build speed? | race |

---

## Domain 2: Football club scout

**Data:** `optimize/behavior_data/all_players.csv` — 17,737 players. Inputs: 55 attributes
(pace, shooting, passing, dribbling, defending, physical ratings; position, age, height,
weight, foot, league). Goals: `Penalties+` (higher is better), `Strength+` (higher is better).

**Client:** the chief scout of a mid-table club. They want a player who can take penalties
under pressure and win physical battles, and their budget pays for very few live scouting trips.

| # | Ask (client's words) | Skill |
|---|---|---|
| 1 | Sort these 17,000 players into the types of player they really are. | remember |
| 2 | Show me the players who are both strong and good from the spot. | remember |
| 3 | Before I send a scout, guess how good this player is at penalties. | guess |
| 4 | Guess his strength from his other numbers, and say how confident you are. | guess |
| 5 | Is our shortlist really better than the league average, or is that luck? | certify |
| 6 | Are strikers actually better penalty takers than midfielders, or does it just look that way? | certify |
| 7 | As scouting reports come in one by one, tell me which players to follow up. | flow |
| 8 | Tell me when this season's new players stop looking like last season's. | flow |
| 9 | Why are some players weak from the spot? What are they missing? | blame |
| 10 | What separates the physically strong players from the rest? | blame |
| 11 | Explain to the manager why we should sign this player. | justify |
| 12 | Why did you rate this player low? Give me reasons I can check on video. | justify |
| 13 | Find the best strong penalty-taker while sending scouts to see only 20 players. | choose |
| 14 | Find the best one under 24, in a league we already scout. | choose |
| 15 | This player is close: what would he need to improve to make our shortlist? | fix |
| 16 | What training focus would turn our weak penalty-taker into a reliable one? | fix |
| 17 | Flag player records that look like data-entry mistakes. | spot |
| 18 | Which players are so unusual we should look twice before trusting their ratings? | spot |
| 19 | Does our shortlist beat a simple "top overall rating" list? | race |
| 20 | Does our penalty guesser beat an off-the-shelf model? | race |

---

## Domain 3: Retail marketing manager

**Data:** `optimize/sales_data/Marketing_Analytics.csv` — 2,205 customers. Inputs: income,
age, household (kids, teens), marital status, education, recency, spend by product
(wine, meat, fish, fruit, sweets, gold), past campaign acceptance. Goals (8): purchases via
deals, web, catalogue and store (all `+`), web visits `+`, complaints `-`, response to the
latest campaign `+`, customer tenure in days `+`.

**Client:** the marketing manager of a retail chain. They want loyal, active customers who
respond to campaigns and don't complain, and each campaign mailing has a cost.

| # | Ask (client's words) | Skill |
|---|---|---|
| 1 | Group our 2,200 customers into the kinds of shopper they really are. | remember |
| 2 | Who are our best customers: busy across channels, loyal, no complaints? | remember |
| 3 | Guess how much a new customer will buy online versus in store. | guess |
| 4 | Guess whether this customer will answer our next campaign. | guess |
| 5 | Did the last campaign really lift purchases, or is that noise? | certify |
| 6 | Do families with teenagers really shop differently, or does it just look that way? | certify |
| 7 | As new customers sign up, tell me straight away who's worth a welcome offer. | flow |
| 8 | Warn me when this month's sign-ups stop looking like our usual customers. | flow |
| 9 | Why do some customers never respond to campaigns? | blame |
| 10 | What do the customers who complain have in common? | blame |
| 11 | Explain to the sales director why this group should get the next campaign. | justify |
| 12 | Why did you mark this customer as low value? Give me reasons I can act on. | justify |
| 13 | Find the best customers to target while test-mailing only 50 people. | choose |
| 14 | If I care most about store visits rather than web orders, who should I target instead? | choose |
| 15 | This customer is drifting away: what's the smallest nudge that brings them back? | fix |
| 16 | Which product offer would turn a deal-only shopper into a regular one? | fix |
| 17 | Flag customer records that look wrong: impossible ages, odd incomes, typos. | spot |
| 18 | Which customers behave unlike anyone else? Are they fraud, or just unusual? | spot |
| 19 | Does our targeting beat simply mailing our highest-income customers? | race |
| 20 | Does our response guesser beat an off-the-shelf model? | race |

---

## Skill coverage

Every skill is demoed twice in every domain:

| Skill | SS-W | all_players | Marketing |
|---|---|---|---|
| remember | 1, 2 | 1, 2 | 1, 2 |
| guess | 3, 4 | 3, 4 | 3, 4 |
| certify | 5, 6 | 5, 6 | 5, 6 |
| flow | 7, 8 | 7, 8 | 7, 8 |
| blame | 9, 10 | 9, 10 | 9, 10 |
| justify | 11, 12 | 11, 12 | 11, 12 |
| choose | 13, 14 | 13, 14 | 13, 14 |
| fix | 15, 16 | 15, 16 | 15, 16 |
| spot | 17, 18 | 17, 18 | 17, 18 |
| race | 19, 20 | 19, 20 | 19, 20 |

## Notes:

- **all_players** contains identifier columns (`num`, `Rank`, `Name`, `url`). Should we use
  them as-is, or may our loader skip them? The file itself would not change.
- **Marketing_Analytics:** a quick check shows that complaints, campaign response and tenure
  are barely predictable. We expect the guess skill to fail on those goals and will report
  that honestly.
- Domains are built in the order above. Domain 3 is the transfer test: how much of the
  domain 1–2 code it reuses.
