# Face Embedding Map

> **Do learned face embeddings separate *identity* better than raw pixels do — and can I show it
> with a number, not just a nice-looking picture?**

This repo answers that question two ways on the same 400 faces, then takes the winning representation
out of the lab and points a webcam at it.

- **Map A — raw pixels / Eigenfaces.** Flatten each 64×64 Olivetti photo to 4096 numbers, run PCA,
  plot. The 1991 method, implemented properly rather than as a strawman.
- **Map B — learned embeddings.** Push the same photos through a *frozen*
  InceptionResnetV1 pretrained on VGGFace2, get a 512-number unit vector per face, run **the same
  PCA with the same settings**, plot the same way.
- **The live study.** 12 webcam scans of my own face × 10 frames, under six lighting conditions
  (room light and phone flash, from front / left / right / below), to see where the embedding moves
  when the only thing changing is the light.

No model is trained anywhere. The network is frozen in `eval()` mode; the comparison is about what
each representation *already* contains.

**The paper:** [`paper/main.pdf`](paper/main.pdf) — method, architecture diagrams, full results,
limitations, and a reproducible protocol. This README is the short version.

---

## The headline

All metrics are computed in **full dimensionality** (4096-d and 512-d), never on the 2-D plots.
400 images, 40 identities, 10 photos each.

| metric | Map A (raw pixels) | Map B (embeddings) |
|---|---|---|
| silhouette (higher is better) | 0.106 | **0.497** |
| 1-NN leave-one-out accuracy | 94.5 % | **100 %** |
| 1-NN top-5 | 98.3 % | **100 %** |
| ROC-AUC, genuine-vs-impostor pairs | 0.881 | **0.9995** |
| EER, pair protocol (lower is better) | 20.5 % | **1.11 %** |
| EER, verification protocol | 8.7 % ± 1.2 | **0.22 % ± 0.24** |
| TAR @ FAR = 10⁻³, pair protocol | 26.4 % | **95.5 %** |
| TAR @ FAR = 10⁻², pair protocol | 42.3 % | **98.8 %** |
| TAR @ FAR = 10⁻³, verification protocol | 59.3 % | **99.4 %** |
| FAR @ FRR = 1 %, verification protocol | 48.6 % | **0.044 %** |
| d′ | 1.61 | **4.99** |

The two protocols must not be conflated. The **pair** protocol scores one probe against one gallery
image. The **verification** protocol scores a probe against the *minimum* distance to five enrolment
images — which is how a face-unlock system actually works — so both maps improve, and the ranking
does not change.

**The one number:** under the verification protocol, at a 0.1 % false-accept rate Map B accepts
99.4 % of genuine probes; raw pixels accept 59.3 %. At a more permissive 1 % false-accept rate it is
99.9 % vs 74.6 %. At a matched 1 % false-*reject* rate, raw pixels let 48.6 % of impostors through;
Map B lets 0.044 % through. The verification error is **38.8× lower** with the embedding.

That gap is the whole project. Raw pixels are not *useless* — 94.5 % of nearest neighbours are still
the right person — but they fail the thing FaceID needs, because illumination moves a face further
than identity does.

---

## The two maps

Same 400 points, same PCA settings (`n_components=2`, randomized SVD, seed 0), same colours for the
same identities, same axis conventions. Only the representation changes.

**Map A — raw pixels:**
![Map A: PCA of raw Olivetti pixels, coloured by identity](figures/mapA_scatter.png)

**Map B — learned embeddings:**
![Map B: PCA of frozen InceptionResnetV1 embeddings, coloured by identity](figures/mapB_scatter.png)

And here is the trap this project is about: **Map B looks *less* clustered than Map A.** That is not
a failure of the embedding; it is a failure of the 2-D projection. I measured how much of the true
geometry each plot preserves (Spearman correlation between 2-D and full-dimensional pairwise
distances): **0.807 for raw pixels, 0.517 for embeddings.** Map A's two lobes are not people: its
first axis correlates **0.986** with the mean brightness of the photo and its second **0.945** with
the left–right brightness gradient. The embedding's identity structure is spread across hundreds of
dimensions, so squashing it to 2-D destroys it. The nicer-looking map is the worse space.

This is the project's central methodological rule: **measure in full dimensions, plot in 2-D, never
measure on the plot.**

The metrics behind the table:

![Genuine vs impostor distance distributions](figures/distance_histograms.png)
![ROC curves](figures/roc_curves.png)

Why the embedding wins is visible in the histograms: for Map B the genuine and impostor distributions
almost separate (EER 1.11 %); for Map A they overlap heavily (EER 20.5 %).

---

## The lighting experiment (one subject, twelve scans)

12 scans × 10 frames = 120 embeddings of me, captured 6 September 2026. Two scans per condition, so
"same condition, different minute" is the noise floor.

![Mean distance between every pair of scans](figures/live_heatmap.png)

![Frame jitter vs lighting effect](figures/live_lighting.png)

What it shows, on the 0–2 Euclidean scale for unit vectors:

- **Frame-to-frame jitter** within one scan: mean **0.186** (max 0.442). This is the floor any
  threshold has to clear.
- **Same condition, second scan**: mean **0.280**. Repeating the capture in the same light is worse
  than consecutive frames, as expected.
- **Light intensity** (room light vs phone flash, both front): mean **0.258** — *inside* the
  repeat-scan noise. Making the face much brighter did not move the embedding.
- **Light direction** (flash only, front/left/right/bottom): mean **0.714**, i.e. **3.83×** the
  frame jitter. Front-lit vs left-lit is 0.71; right-lit vs bottom-lit reaches **0.967**, the single
  worst same-person distance in the dataset.
- **Under-lighting is the worst case.** The bottom-lit baseline is also the noisiest, and bottom-lit
  faces are the furthest from everything else. Faces are rarely lit from below in training photos.
- **Frames are not independent.** Distance grows with frame lag: 0.153 between consecutive frames,
  0.257 between the first and last frame of a scan. Small head/pose drift happens during the ~2 s
  capture, so ten frames are not ten independent samples. Frame 0 is an outlier (0.243 mean distance
  vs 0.167 at frame 5) — which is why `scripts/live_capture.py` now throws away warm-up frames.
- **Enrolment trick:** averaging the 10 frames of a scan before re-normalising reduces across-scan
  distances by only **4.8 %** — small, because distance is dominated by lighting, not sensor noise.

**Where does a threshold go?** Any threshold below **0.967** rejects me on my own data. That is the
convenience side of the security dial, and it alone is why a naive "cosine distance < 0.5" rule fails
the moment the light moves.

**No stranger scan exists yet.** Every number in this section is me-vs-me. A cross-domain sanity
check (distance from my scans to the 400 Olivetti embeddings) gives a typical nearest-stranger
distance of **1.288** but a minimum of **0.887** — the tails overlap the genuine range, and the
comparison is confounded anyway (different capture, crop and resolution). The calibrated
genuine/impostor analysis therefore comes from the controlled Olivetti protocol, not from the
webcam. This is the biggest limitation in the repo.

---

## What this means for FaceID

A FaceID-style system is a distance threshold: below `X` accept, above `X` reject. The project makes
you feel the trade-off instead of reading about it:

- Tighten `X` and your own dim/bottom-lit photo gets rejected (FRR rises).
- Loosen `X` and a stranger gets in (FAR rises).
- The EER is where those two error rates meet. On Olivetti, with the same enrolment/probe protocol,
  raw pixels cross at **8.7 %**, and the frozen embedding crosses at **0.22 %**. At a 0.1 % FAR the
  embedding still accepts 99.4 % of genuine probes; raw pixels accept 59.3 %. At a 1 % FAR it is
  99.9 % vs 74.6 %.
- The threshold is **domain-specific**. A number fitted on Olivetti's tightly cropped, aligned,
  greyscale faces does not transfer to a webcam frame without recalibration — and that is exactly
  the kind of thing that is easy to get wrong when quoting a single benchmark number.

---

## What I got wrong (and what I fixed)

- **I had a claim, not a result.** Map A was finished enough to look at but the 2-D scatter, the
  silhouette score and the Map B comparison — the actual experiment — were still "to do" after the
  first write-up. This build finishes them.
- **I eyeballed the eigenfaces, twice.** I guessed 6 components mattered, then 20–30 after looking at
  the scree plot. The measurement said 19 components for 75 % of variance and 66 for 90 %. The flat
  tail is not empty: components 20–66 carry another 15 %.
- **I called the model "robust to lighting".** That is only half true. Intensity is free; direction
  is expensive — 3.8× the frame noise. "Robust to lighting" is not a property, it is a direction.
- **I under-reported my own worst case.** The findings quoted 0.94 as the worst same-person distance;
  the data says 0.967. Small, but it moves the lowest workable threshold.
- **I talked about a "noise floor" without measuring the within-scan floor.** The original number was
  between-scan variation. There are two floors (0.186 within a scan, 0.280 between scans), and a
  threshold has to clear both.
- **I mixed up Euclidean and cosine naming.** The live distances are Euclidean between unit vectors
  (`d² = 2 − 2cos`). The plan said "cosine is standard" while the code used Euclidean. Same ranking,
  different scale — dangerous when comparing with other papers. Now both are reported.
- **PCA was quietly non-deterministic.** scikit-learn picks the randomized SVD solver for this data,
  and the notebook left `random_state=None`: identical cells gave explained-variance ratios differing
  by up to 1.8×10⁻⁵. Fixed with `seed=0`.
- **The pipeline existed twice.** `scripts/compare_scans.py` and the notebook each had their own data
  loading and distance code; a third copy was coming for Map B. Now one library (`fem/`) does each
  stage once.
- **I forgot the webcam warms up.** The first frames are black (mean brightness 3/255), so early
  frames silently produce no face. They are now discarded, and the data confirms frame 0 is the
  outlier.
- **I fit PCA on the probe images.** Transductive, and it cannot hurt Map A — so the Map B result is
  conservative. Stated rather than hidden.
- **I committed almost nothing.** At the start of this pass, `git` tracked a PyCharm `main.py`
  boilerplate and one notebook; none of the scans, scripts, findings or plans were in the repo. Fixed.

---

## Reproduce it

Everything runs in the existing project virtualenv (`~/.venvs/nn-crew`; the versions are pinned in
`requirements.txt`). Model weights and the Olivetti data are cached locally. `make help` lists every
target.

```sh
make test        # 61 pytest tests, offline, headless, < 60 s
make all         # tests + all four experiment runners (~25 s; deterministic)
make diagrams    # rebuild the paper's architecture diagrams from TikZ sources
make paper       # compile paper/main.pdf (two passes)
make notebook    # execute notebooks/01-map-a-raw-pixels.ipynb in place
```

The underlying commands, if you prefer running them directly:

```sh
export MPLCONFIGDIR="$PWD/.cache/mpl"
PY=/Users/nic/.venvs/nn-crew/bin/python

$PY analysis/run_olivetti.py    # Map A + Map B + all metrics -> results/olivetti.json, figures/
$PY analysis/run_live.py        # lighting study           -> results/live.json
$PY analysis/run_threshold.py   # open-set verification     -> results/threshold.json
$PY analysis/model_arch.py      # verified layer shapes     -> results/model_arch.json
$PY -m pytest -q                # library tests, offline, < 60 s

cd paper && pdflatex -interaction=nonstopmode main.tex && \
            pdflatex -interaction=nonstopmode main.tex   # -> paper/main.pdf
```

Every number that reaches the paper is written to `results/*.json` by one of those scripts — no
hand-typed numbers anywhere.

Live capture (needs a webcam and camera permission; writes into `data/scans/`):

```sh
$PY scripts/live_capture.py --intensity flash --direction left --repeat 3 --frames 10
```

## Repository map

```
fem/            the library — one module per pipeline stage
                config · data · preprocess · embed · maps · metrics · plotting
analysis/       experiment runners (E1–E5); each writes results/*.json + figures/
scripts/        live_capture.py (webcam CLI) · compare_scans.py (distance report)
data/           scans.json manifest · scans/ (12 raw scans) · README · sklearn cache (ignored)
results/        machine-readable results (JSON, source of truth) + caches (npz, ignored) + README
figures/        result figures, PNG (200 dpi) + vector PDF, + README
notebooks/      01-map-a-raw-pixels.ipynb (cleaned, drives fem)
paper/          LaTeX paper, section sources, architecture diagrams, compiled main.pdf
docs/           DESIGN · REVIEW · CODE · ROADMAP · PLAN · findings/
tests/          pytest suite
```

## Read more

- **Paper:** [`paper/main.pdf`](paper/main.pdf) — 16 pages; method, diagrams, results, limitations.
- **Live-study notes:** [`docs/findings/live-illumination.md`](docs/findings/live-illumination.md) (Observed / Means / So what)
- **Map A notes:** [`docs/findings/map-a-raw-pixels.md`](docs/findings/map-a-raw-pixels.md)
- **Notebook walkthrough:** [`notebooks/01-map-a-raw-pixels.ipynb`](notebooks/01-map-a-raw-pixels.ipynb)
- **Next steps:** [`docs/ROADMAP.md`](docs/ROADMAP.md)
- **How the code fits together:** [`docs/CODE.md`](docs/CODE.md)
- **The original plan and its traps list:** [`docs/PLAN.md`](docs/PLAN.md)
- **Why the repo looked the way it did / what was fixed:** [`docs/REVIEW.md`](docs/REVIEW.md)

## Limitations, plainly

- **One subject, no stranger captures.** The live lighting study measures within-identity
  variability; it cannot set a threshold.
- **Olivetti is pre-aligned and tightly cropped.** Raw pixels get their best-case data here; LFW or a
  webcam would be much harder for Map A.
- **400 images, frozen model.** No training, so no claim about fine-tuning; the CNN has already seen
  millions of faces, which is part of why it wins.
- **Webcam frames are temporally correlated** and the 10 frames per scan are not independent.
- **Cross-domain numbers are not calibrated.** Olivetti ↔ webcam distances are an upper bound on
  stranger distance, not an operating threshold.
- **Verification uses 5 random 5/5 identity splits**; the fixed first-half/second-half split is
  reported only as a session-bias check.

## Next

1. Capture strangers under the same six lighting conditions — the missing negative class.
2. **Yale Face Database B** (Stage 6) for a controlled illumination-direction experiment on real
   faces, not one phone flash.
3. **LFW** for an unconstrained benchmark with the same frozen model.
4. **Triplet-loss training** (Stage 7) — the only stage where training would mean something.

---

*Data captured 6 September 2026. Analysis and paper rebuild: October 2026.*
