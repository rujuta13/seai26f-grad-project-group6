PY ?= python3

.PHONY: data test loc step1 step2 step3 step4 step5

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
