PY ?= python3

.PHONY: data test loc step1 step2 step3 step4 step5 step6 step7 step8 step9 step10 step11 step12 step13 step14 step15 step16 step17 step18 step19 step20 step21 step22 step23 step24

data:            ## download the three MOOT csvs (unmodified)
	bash data/get.sh

test: data       ## run every step's tests
	$(PY) -m pytest -q tests

loc:             ## new lines of code per step tag
	$(PY) harness/loc.py

# One target per step is added as each step lands.
step1: data      ## remember: families of flag sets, SS-W
	PYTHONPATH=src $(PY) -m domains.ssw.step1

step2: data      ## guess: speed and CPU of an unbenchmarked flag set, SS-W
	PYTHONPATH=src $(PY) -m domains.ssw.step2

step3: data      ## certify: is a gap real or noise, SS-W
	PYTHONPATH=src $(PY) -m domains.ssw.step3

step4: data      ## flow: keep or warn as results arrive, SS-W
	PYTHONPATH=src $(PY) -m domains.ssw.step4

step5: data      ## blame: what makes bad builds bad, SS-W
	PYTHONPATH=src $(PY) -m domains.ssw.step5

step6: data      ## justify: checked reasons for a recommendation, SS-W
	PYTHONPATH=src $(PY) -m domains.ssw.step6

step7: data      ## choose: best build from 30 benchmarks, SS-W
	PYTHONPATH=src $(PY) -m domains.ssw.step7

step8: data      ## fix: fewest flips from slow to fast, SS-W
	PYTHONPATH=src $(PY) -m domains.ssw.step8

step9: data      ## spot: glitches and oddballs, SS-W
	PYTHONPATH=src $(PY) -m domains.ssw.step9

step10: data     ## race: our tuner and guesser vs rivals, SS-W
	PYTHONPATH=src $(PY) -m domains.ssw.step10

step11: data     ## remember (players): types of player and the best 1%
	PYTHONPATH=src $(PY) -m domains.players.step11

step12: data     ## guess (players): penalties and strength from other ratings
	PYTHONPATH=src $(PY) -m domains.players.step12

step13: data     ## certify (players): is a gap real or luck
	PYTHONPATH=src $(PY) -m domains.players.step13

step14: data     ## flow (players): keep or warn as reports arrive
	PYTHONPATH=src $(PY) -m domains.players.step14

step15: data     ## blame (players): what weak penalty-takers lack, what strong players have
	PYTHONPATH=src $(PY) -m domains.players.step15

step16: data     ## justify (players): reasons to sign or rate low
	PYTHONPATH=src $(PY) -m domains.players.step16

step17: data     ## choose (players): best player from 20 scouting trips
	PYTHONPATH=src $(PY) -m domains.players.step17

step18: data     ## fix (players): what a player must improve
	PYTHONPATH=src $(PY) -m domains.players.step18

step19: data     ## spot (players): doubtful ratings and unusual players
	PYTHONPATH=src $(PY) -m domains.players.step19

step20: data     ## race (players): our shortlist and guesser vs rivals
	PYTHONPATH=src $(PY) -m domains.players.step20

step21: data     ## remember (marketing): kinds of shopper and the best customers
	PYTHONPATH=src $(PY) -m domains.marketing.step21

step22: data     ## guess (marketing): channels and campaign response
	PYTHONPATH=src $(PY) -m domains.marketing.step22

step23: data     ## certify (marketing): is a gap between customers real
	PYTHONPATH=src $(PY) -m domains.marketing.step23

step24: data     ## flow (marketing): welcome offers and sign-up drift
	PYTHONPATH=src $(PY) -m domains.marketing.step24
