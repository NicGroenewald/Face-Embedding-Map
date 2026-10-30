# Face Embedding Map — one entry point for every pipeline stage.
#
# Why a Makefile: the commands need the project virtualenv and a writable
# matplotlib cache. Retyping either by hand is how a run happens in the wrong
# environment and nobody notices until the numbers are wrong.
#
# Override the interpreter if needed:  make PY=/path/to/python test

PY ?= /Users/nic/.venvs/nn-crew/bin/python
MPLCONFIGDIR := $(CURDIR)/.cache/mpl
export MPLCONFIGDIR

# Architecture diagrams, compiled from paper/figures/src/ into paper/figures/.
DIAGRAMS := fig_pipeline fig_preprocess fig_model fig_protocol fig_lighting_setup

.PHONY: all help cache test maps live threshold arch compare diagrams paper notebook clean distclean

all: test maps live threshold arch            ## tests + every results-producing script

help:                                          ## list targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*##"}; {printf "  %-12s %s\n", $$1, $$2}'

cache:                                         ## create the writable matplotlib cache
	mkdir -p "$(MPLCONFIGDIR)"

test: cache                                    ## pytest suite (offline, headless, < 60 s)
	$(PY) -m pytest -q

maps: cache                                    ## E1+E2: Map A vs Map B on Olivetti
	$(PY) analysis/run_olivetti.py

live: cache                                    ## E3: webcam illumination study
	$(PY) analysis/run_live.py

threshold: cache                               ## E4: open-set verification / threshold analysis
	$(PY) analysis/run_threshold.py

arch: cache                                    ## E5: verified InceptionResnetV1 layer shapes
	$(PY) analysis/model_arch.py

compare: cache                                 ## human-readable scan-distance report
	$(PY) scripts/compare_scans.py

diagrams:                                      ## rebuild the paper's architecture diagrams
	cd paper/figures/src && for f in $(DIAGRAMS); do \
		pdflatex -interaction=nonstopmode -halt-on-error $$f.tex >/dev/null && mv $$f.pdf ../ ; \
	done

paper:                                         ## compile paper/main.pdf (two passes)
	cd paper && pdflatex -interaction=nonstopmode main.tex
	cd paper && pdflatex -interaction=nonstopmode main.tex

notebook: cache                                ## execute the exploration notebook in place
	$(PY) -m nbconvert --to notebook --execute --inplace notebooks/01-map-a-raw-pixels.ipynb

clean:                                         ## remove build artefacts (keeps data and results)
	rm -rf .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc paper/*.synctex.gz
	rm -f paper/figures/src/*.aux paper/figures/src/*.log paper/figures/src/fig_*.pdf
	rm -rf "$(MPLCONFIGDIR)"

distclean: clean                               ## also drop regenerable caches (Olivetti + embeddings)
	rm -rf data/sklearn
	rm -f results/*.npz
	@echo "Caches removed. 'make all' will re-fetch Olivetti (~4 MB) and re-embed."
