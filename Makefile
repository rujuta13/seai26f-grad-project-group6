PY ?= python3

.PHONY: data test loc step1 step2

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
