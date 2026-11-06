PY ?= python3

.PHONY: data test loc

data:            ## download the three MOOT csvs (unmodified)
	bash data/get.sh

test: data       ## run every step's tests
	$(PY) -m pytest -q tests

loc:             ## new lines of code per step tag
	$(PY) harness/loc.py

# One target per step is added as each step lands, e.g.
# step1: data
# 	$(PY) -m domains.ssw.step1
