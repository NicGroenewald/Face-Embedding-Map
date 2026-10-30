# Face Embedding Map — one entry point for every pipeline stage.
#
# Why a Makefile: DESIGN.md §10 commands must be run with the project venv and
# a writable matplotlib cache.  Retyping either by hand is how a run happens
# in the wrong environment and nobody notices until the numbers are wrong.
#
# Override the interpreter if needed:  make PY=/path/to/python test

PY ?= /Users/nic/.venvs/nn-crew/bin/python
MPLCONFIGDIR := $(CURDIR)/.cache/mpl
export MPLCONFIGDIR

.PHONY: all cache test maps live threshold arch compare paper clean

# Full pipeline: tests first (they are fast and catch broken invariants before
# an hour of analysis), then every results-producing script.
all: test maps live threshold arch

cache:
	mkdir -p "$(MPLCONFIGDIR)"

test: cache
	$(PY) -m pytest -q

# E1 + E2: Map A vs Map B on Olivetti -> results/olivetti.json + figures/
maps: cache
	$(PY) analysis/run_olivetti.py

# E3: live lighting study -> results/live.json + figures/
live: cache
	$(PY) analysis/run_live.py

# E4: open-set verification / threshold analysis -> results/threshold.json
threshold: cache
	$(PY) analysis/run_threshold.py

# E5: verified InceptionResnetV1 layer shapes -> results/model_arch.json
arch: cache
	$(PY) analysis/model_arch.py

# Human-readable sanity check of the saved scans (no model run).
compare: cache
	$(PY) compare_embeddings.py

# Paper: two passes so cross-references resolve.
paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex
	cd paper && pdflatex -interaction=nonstopmode main.tex

clean:
	rm -rf .pytest_cache
	rm -rf $(MPLCONFIGDIR)
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc paper/*.synctex.gz
