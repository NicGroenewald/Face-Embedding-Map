# Repository Review — Face Embedding Map

*Audit date: 30 October 2026. Reviewer: Lead. Scope: the repository as it stood at commit
`51e1f07` plus the uncommitted working tree (12 saved scans, notebook, findings drafts, scripts).*

> **Path note.** This is a historical audit: paths such as `live.py`, `compare_embeddings.py`,
> `saved-face-embeddings/`, `Map A - Raw Pixels/`, `findings-live-embeddings.md` and
> `face-embedding-map.md` refer to the repository *as audited*, before the restructure. They now
> live at `scripts/live_capture.py`, `scripts/compare_scans.py`, `data/scans/`, `docs/findings/`,
> `docs/PLAN.md` and `notebooks/` — see `docs/DESIGN.md` §3 for the current layout.

This document records what is wrong, why it matters, and what this build does about it. It is split
into **science**, **code**, **documentation**, **reproducibility** and **git/repo hygiene**. Each
finding has an evidence pointer and a status:

- **Fixed** — addressed in this build (see `docs/DESIGN.md` for the tasks that own it).
- **Open** — not addressed now; recorded as future work.
- **Not a defect** — checked and found sound; noted so it is not "fixed" into being worse.

The project's own plan is good — the failure mode here is not a bad idea, it is a good idea whose
evidence was never finished or committed. The biggest gap is that **the central experiment (Stages 3
and 4) had never been run**, so the repo had a claim but no number.

---

## 0. One-paragraph verdict

The repository contains real work and honest reasoning, but at HEAD it tracks three files
(`.gitignore`, `main.py`, `data_explore.ipynb`), and `main.py` is PyCharm boilerplate. Every piece
of actual work — the second notebook location, `live.py`, `compare_embeddings.py`,
`requirements.txt`, both findings documents, and all 12 scan files — is uncommitted or mis-staged.
The science stops exactly where it gets interesting: Map A exists but has no 2-D scatter and no
metric; Map B does not exist; the Stage 4 comparison and the Stage 5 threshold analysis were never
done. The live lighting study is the strongest original content and its largest number is
misreported. Fixing the evidence, the git history and the documentation matters more than any code
style change; the code fixes are small and mostly about not parsing the same scan twice in two
inconsistent ways.

---

## 1. Science and methodology

### 1.1 Stage 3 and Stage 4 were never run (critical, Fixed)

The plan's entire point is the number comparison "do learned embeddings separate identity better
than raw pixels". `Map A - Raw Pixels/findings.md` ends with "Still to do: the 2-D scatter ... is not
in the notebook yet. Silhouette score ... is not computed yet. This is the number Map B gets compared
against." Map B does not exist anywhere in the repo. There is no `InceptionResnetV1` use on Olivetti,
no silhouette, no ROC, no EER.

**Why it matters:** without Stages 3–4 the repo has a hypothesis, not a result, and the paper has no
experiment.

**Fixed:** `analysis/run_olivetti.py` (E1/E2), `analysis/run_threshold.py` (E4); results in
`results/olivetti.json`, `results/threshold.json`; figures in `figures/`.

### 1.2 Stage 5 has no negative data (major, Open/partly mitigated)

Every one of the 120 live embeddings is the author. There is no stranger capture, so no impostor
distribution exists in the live domain. `findings-live-embeddings.md` is explicit about this ("Every
number above is me vs me"), but the consequence is larger than the doc admits: **no live threshold
can be set from this data at all**. A threshold drawn from genuine distances alone is guaranteed to
accept the user; whether it accepts a stranger is unmeasured.

**Fixed partly:** the open-set verification experiment (E4) is run on Olivetti, a controlled dataset
with 40 identities, and gives a real FAR/FRR/EER comparison between Map A and Map B. The live study
is kept as a *within-identity variability* experiment. The paper must state that the two are not
the same measurement and must not transpose an Olivetti threshold onto a webcam.

**Open:** capture 1–3 other people under the same six conditions. This is the missing experiment.

### 1.3 The worst live distance is misreported (major, Fixed)

`findings-live-embeddings.md` §5 states the worst me-vs-me distance is **0.94** ("Flash right vs
flash bottom: 0.80 to 0.94, the largest distance measured") and "0.94 is the worst me-vs-me distance
found so far".

Recomputed over all 12 scans × 100 frame pairs, the maximum is **0.967** (flash right repeat 2 vs
flash bottom repeat 2); the right-repeat-1 vs bottom-repeat-2 pair is 0.947. The doc understates the
worst case by ~0.03 and, more importantly, understates the threshold consequence: *any threshold
below 0.967 rejects the enrolled user on his own data*.

**Fixed:** all numbers in the rewritten findings and paper come from `results/live.json`.

### 1.4 Distance ranges in the findings are approximate, and one is mislabelled (major, Fixed)

Checked every range in `findings-live-embeddings.md` against the scan files:

| claim | recomputed | verdict |
|---|---|---|
| same-condition repeats 0.18–0.37 (room front) | 0.187–0.374 | accurate |
| same-condition repeats 0.18–0.36 (flash front) | 0.184–0.355 | accurate |
| same-condition repeats 0.18–0.34 (flash right) | 0.176–0.339 | accurate |
| side/bottom repeats 0.23–0.50 (flash left), 0.25–0.47 (bottom) | 0.228–0.497, 0.250–0.469 | accurate |
| room front vs flash front 0.20–0.38 | 0.168–0.384 | min understated |
| flash front vs flash left 0.70–0.80 | 0.620–0.834 | range too narrow |
| flash front vs flash bottom 0.63–0.79 | 0.604–0.835 | range too narrow |
| flash left vs flash right 0.44–0.62 | 0.418–0.620 | close |
| flash right vs flash bottom 0.80–0.94 | 0.795–0.967 | max wrong (see 1.3) |

The narrative conclusions survive — direction moves the embedding several times more than repeat
jitter, intensity does not — but the ranges are not what a paper can quote.

**Fixed:** recomputed in `analysis/run_live.py`; findings rewritten from the JSON.

`compare_embeddings.py` also labels `room_front_1` vs `room_left_1` as "mild lighting change, from
earlier" (line 41). It is not the earlier mild test — that test is discussed in §6 of the findings
and used a different pair of photos. Room-front vs room-left is a **direction** change within the
room-light condition. The comment is inherited confusion, not a data problem.

A second comparison in the same list, `room_front_1` vs `flash_bottom_1`, appears in the findings as
"room front vs flash bottom 0.63–0.79" — but it changes *intensity and direction at once*, so it
cannot support a lighting-direction claim. It is not wrong as a measured distance, only as evidence;
the rewrite drops it from the direction discussion.

### 1.5 The "noise floor" mixes two different noise sources (major, Fixed)

§1 of the live findings uses distances between the two *scans* of a condition ("two scans a minute
apart") and calls that the jitter "any threshold has to sit above". The within-scan frame-to-frame
spread is never reported, even though the data supports it (45 pairs per scan). It is larger than
the doc implies: within-scan maxima reach **0.442** (flash bottom repeat 1; mean within-scan
distance 0.334), above the "threshold below 0.4 rejects me" line the doc draws.

There are therefore two noise floors — frame-to-frame within one second, and scan-to-scan across a
minute — and they differ. The paper reports both.

**Fixed:** `within_scan_jitter` and `same_condition` are separate keys in `results/live.json`.

### 1.6 PCA nondeterminism (minor, Fixed)

The notebook calls `PCA(n_components=100)` with defaults. scikit-learn selects the **randomized SVD**
solver here (400×4096, `n_components=100 < 0.8·min(400, 4096)`, `max(shape)=4096 > 500`) and
`random_state=None`. Re-running the same notebook cell three times gives `explained_variance_ratio_`
differences up to **1.8×10⁻⁵** and slightly different eigenvectors. Small, but it means the
eigenface images and any threshold-dependent milestone in the notebook are not exactly reproducible.

**Fixed:** `fem.maps.pca_map(..., seed=0)` passes `random_state=0`; `random_state` is recorded in
`results/olivetti.json`.

### 1.7 Silhouette across spaces of different dimension (major, Fixed by control)

Map A is 4096-d, Map B is 512-d. Silhouette, and any Euclidean distance distribution, is affected by
dimension (distance concentration), so a naked silhouette comparison is not airtight. The plan's
instruction to compare them is still the right experiment, but the paper needs a matched-dimension
control or a reviewer will (fairly) object.

**Fixed:** `results/olivetti.json → controls.map_a_pca512` reruns the identical metrics on Map A
reduced to 512 PCA components. Rank-based metrics (1-NN, ROC-AUC, EER) are also reported because
they are less sensitive to this.

### 1.8 Map A is fit on the probe data (minor, disclosed)

PCA for Map A is fit on all 400 images, including the images later used as probes. That is
transductive and cannot hurt Map A. Since Map B is fully frozen, any Map B advantage is therefore
conservative. This is not an error — the plan explicitly says "no train/test split" — but it must be
stated in the paper (it is now recorded in `results/olivetti.json → notes`).

### 1.9 Metric naming: Euclidean is used, cosine is claimed (minor, Fixed)

`findings-live-embeddings.md` says distances are Euclidean between unit vectors (correct: `d² = 2 −
2cos`), while `face-embedding-map.md` Stage 5 says "Cosine distance is standard for these". Both are
defensible; using both names without stating the conversion is not. The two are rank-equivalent for
unit vectors, but the *numbers* live on different scales (Euclidean 0–2, cosine 0–1), and a reader
comparing against a paper that quotes cosine will be confused.

**Fixed:** Euclidean is the primary metric everywhere (so the existing live numbers stay valid), and
cosine equivalents are emitted alongside (`cos = 1 − d²/2`) in JSON and in the findings.

### 1.10 Claims that outweigh the data (minor, Open)

"Robust to lighting" appears in the live findings as a half-true claim about the model. It is a
useful summary but it is an interpretation, not a measurement; the paper must attribute it to this
specific model and this specific data (one subject, six conditions). Likewise "the model appears to
have a half-shadowed region in its space" (§4) is a hypothesis from 12 scans, not an established
property. Rewritten findings label these as hypotheses.

**Not a defect:** the honest "an earlier mild test was confounded" section (§6) is exactly the right
instinct and is kept.

---

## 2. Code

### 2.1 Two inconsistent copies of the pipeline (major, Fixed)

`compare_embeddings.py` hardcodes 12 filenames in a dict (lines 4–17) and reimplements pairwise
distance loops internally (lines 24–30). `live.py` hardcodes the output directory and a 10-frame
scan. There is no shared loader, no shared distance function, and `analysis/` does not exist yet, so
every future experiment would grow a third copy. The project's own trap #4 ("don't build the
pipeline twice") was already being violated.

**Fixed:** all pipeline stages live in `fem/` (`data`, `preprocess`, `embed`, `maps`, `metrics`,
`plotting`); scripts import them.

### 2.2 Scan metadata is encoded only in filenames (major, Fixed)

The 12 scans are identified by `06092026-light_face_front_1.npy` etc. Nothing records subject,
condition semantics, frame count, model, device or capture date inside the file. Renaming a file
silently changes a condition; there is no way to tell from the data what "phone_light" meant.
`compare_embeddings.py` proves the problem by carrying a hand-maintained name→file map, and
`live.py` does not let the user name a scan at all — it writes `ddmmyyyy-HHMMSS.npy` only, so every
condition label in the repo was applied by hand afterwards.

**Fixed:** `data/scans.json` is the single source of truth for scan → file + condition metadata;
`fem.data.load_scans()` reads it; `live.py` accepts `--label`/`--intensity`/`--direction` and writes
the same schema.

### 2.3 `live.py` bugs and fragility (major, Fixed)

| line | issue |
|---|---|
| 13, 39, 45 | `mtnn(rbg)` then `mtnn.detect(rbg)` — the detector runs twice per frame for no benefit |
| 34–35 | inner loop does not check `ret`; a camera failure propagates as a confusing `cvtColor` error |
| 18–20 | no warm-up: the first frames are black (documented in the findings: mean brightness 3/255) and yield "no face found", so a 10-frame scan can silently save fewer than 10 rows |
| 38 | variable typo `rbg` (cosmetic, but it is the kind of thing that hides the next bug) |
| 69 | `cv2.destroyWindow("Live Face tracking")` raises if the window was never created (e.g. no face in any frame) |
| 71–72, 79–81 | no `try/finally`: an exception leaves the camera and windows open |
| 78 | timestamp to the second: two scans started within the same second silently overwrite each other |
| 74–84 | saves only the raw matrix; no metadata, no model name, no device, no condition |
| — | module-level code: importing the file opens the webcam; no `main()`, no argparse |
| 60–63 | `.cpu().numpy()` is fine; no unit-norm assertion, so a preprocessing bug would go unnoticed |
| — | MTCNN on CPU while the embedder is on MPS is correct (and undocumented in code) — keep it, with a comment |

**Fixed:** `live.py` refactored with argparse, warm-up frames, single detection per frame, metadata,
collision-safe filenames, `try/finally`, and explicit device comments.

### 2.4 `compare_embeddings.py` issues (major, Fixed)

| line | issue |
|---|---|
| 4–17 | hardcoded file map; adding a scan requires editing code |
| 24–30 | Python double loop instead of vectorised distances; no cosine equivalents, no unit-norm check |
| 41 | mislabelled comparison (see §1.4) |
| 51–54 | compares only repeat `_1` for several conditions, discarding repeat `_2` and the within-scan pairs |
| 56–57 | self-check is `assert ... == 0` on floats; works only because the same array is compared to itself, and asserts vanish under `python -O` |
| — | no scan-mean/pooling analysis, which is the actionable result for enrolment |

**Fixed:** `compare_embeddings.py` now loads from the manifest, vectorises, reports Euclidean and
cosine, adds within-scan jitter, and keeps the self-check as a pytest test rather than an assert.

### 2.5 No tests, no seeds, no entry points (major, Fixed)

There is no test anywhere, no `if __name__ == "__main__"` guard, no CLI, no config module, no
`pyproject.toml`, no `Makefile`. Nothing pins the seed; the notebook is the only executable record
of Map A, and it depends on a download that is cached at a different absolute path on the author's
machine (`/root/scikit_learn_data` in the cell output — i.e. run in a container/Colab).

**Fixed:** `tests/` (metrics, maps, data, preprocessing), `fem/config.py` (paths + seed), argparse
CLIs, `pyproject.toml`, `Makefile`.

### 2.6 Dependency pinning is real but unverified (minor, Fixed)

`requirements.txt` was generated from imports and pins exact versions, which is good. It is not
installed by any documented command in the repo (`uv pip install -r requirements.txt` exists only as
a comment), and there is no `pyproject.toml` or lockfile. Anaconda's environment and the project venv
differ by several minor versions (e.g. scikit-learn 1.7 vs 1.9), so "it works on my machine" is
currently unverifiable from the repo alone.

**Fixed:** `Makefile`, README install instructions, versions recorded in every `results/*.json`.

---

## 3. Documentation

### 3.1 There is no README (critical, Fixed)

The repository has no `README.md`. A visitor sees a PyCharm sample file, a notebook and nothing
else. The plan's own closing instruction — "The repo README should open with the question and close
with the two numbers from Stage 4" — was never carried out.

**Fixed:** `README.md` written last, after the numbers exist.

### 3.2 The project plan is gitignored (critical, Fixed)

`.gitignore` line 7 is `face-embedding-map.md`. The single best document in the repo — the staged
plan with its traps list — is deliberately invisible to git. This looks like a mistaken attempt to
ignore a scratch note and is the most damaging single line in the repo.

**Fixed:** removed from `.gitignore`; the plan is committed.

### 3.3 Findings documents are drafts and are not linked (major, Fixed)

Both findings files start with "*Draft for rewriting*". They are good lab notes but they are not a
write-up: numbers are approximate (§1.4), the biggest claim is wrong (§1.3), there is no method
section, no figures referenced, and the notebook findings are duplicated in two places. There is no
statement of what a reader should conclude.

**Fixed:** both rewritten; the paper is the citable write-up; README links them.

### 3.4 Notebook hygiene (minor, Fixed)

`Map A - Raw Pixels/data_explore.ipynb`:

- imports `filecmp.cmp` (unused) and `StandardScaler`, then computes `X_scaled = ...` (unused
  compute, and the findings explicitly reject scaling — a contradiction left in code),
- has an empty trailing cell,
- never plots the 2-D scatter that Stage 2 requires,
- embeds four PNGs, making the notebook 4.1 MB and the outputs unreproducible from the source alone,
- hardcodes no data path, so it re-downloads to `~/scikit_learn_data` (or fails in a sandbox),
- leaves the dataset in a global `dataset` and defines no functions — nothing is reusable,
- takes the first 100 eigenfaces in a 10×10 grid, a figure that is not readable at any size.

**Fixed:** notebook trimmed, dead code removed, 2-D scatter added via `fem`, figures moved to
`figures/`, outputs regenerated, size reduced.

### 3.5 Missing standard repo documents (minor, partly fixed)

No `LICENSE`, no `CITATION.cff`, no CONTRIBUTING, no `.editorconfig`. For a project whose stated
output is a paper, `CITATION.cff` and a license are cheap and expected. Not blocking; noted as
backlog (choose a license deliberately — do not add MIT silently if the intent is different).

### 3.6 The paper never existed (critical, Fixed)

The plan's "Writing it up" section was never started. No LaTeX, no figures, no references.

**Fixed:** `paper/main.tex` + sections + architecture diagrams + result figures, compiled PDF.

---

## 4. Reproducibility

### 4.1 Most work is uncommitted (critical, Fixed)

At HEAD (`51e1f07`) the repository tracks **three files**:

```
.gitignore
main.py                 # PyCharm "print_hi" boilerplate
data_explore.ipynb
```

None of the following is tracked: `live.py`, `compare_embeddings.py`, `requirements.txt`,
`findings-live-embeddings.md`, `Map A - Raw Pixels/findings.md`, all 12 `saved-face-embeddings/*.npy`
files. The index is worse than the working tree: `git status` shows staged additions of
`Map A - Raw Pixels/data_explore.ipynb`, deletions of the root notebook and `main.py`, an
`AM compare_embeddings.py`, and an `AD saved_embeddings` entry — a zero-byte file staged, then
deleted from disk. `git ls-files -s` shows `compare_embeddings.py` and `saved_embeddings` staged at
blob `e69de29` (the empty blob). Nobody can reproduce, review or cite this work from the repository.

**Fixed:** clean, logical commit history at the end of this build (Lead commits; see README
"Repository state").

### 4.2 No data provenance (major, Fixed)

The 12 scan files are the only record of the lighting experiment, and they carry no capture
metadata. `findings-live-embeddings.md` says "taken 6 Sep 2026", but nothing in the data says so, and
the framing conditions (camera distance, phone position) exist only in prose. The report also does
not give the detector/embedder versions or the device.

**Fixed:** manifest + per-scan metadata + versions/device/hardware recorded in every results JSON;
the paper has a reproducibility section.

### 4.3 The Olivetti cache path is container-specific (minor, Fixed)

Notebook output shows `downloading Olivetti faces ... to /root/scikit_learn_data`, which is a
Linux/container home directory, not this Mac. Anyone re-running the notebook without network gets a
download attempt or failure; the data is not in the repo and not cached at a known path.

**Fixed:** `data/sklearn` is the configured `data_home`, created by `load_olivetti()`; the repo
documents that it is a cache, not source data.

### 4.4 No CI (minor, Open)

No GitHub Actions, no lint, no test run on push. With a test suite now present, a single workflow
running `pytest` and `python -c "import fem"` would be cheap. Left as backlog (needs a repo decision
on GitHub Actions usage).

---

## 5. Git / repository hygiene

| finding | evidence | status |
|---|---|---|
| Plan document intentionally ignored | `.gitignore:7` | Fixed |
| PyCharm boilerplate `main.py` committed | `HEAD:main.py` (print_hi) | Fixed (removed in integration) |
| Empty blob staged as `saved_embeddings` | `git ls-files -s` → `e69de29` | Fixed |
| Root notebook deleted, copy staged under `Map A - Raw Pixels/` | `git status` | Fixed |
| Index has both `A` and `D` entries for the same move | `git status` | Fixed |
| No line-ending/attribute normalisation | no `.gitattributes` | Open (minor) |
| Large binary-ish notebook embedded images | 4.1 MB file | Fixed (figures regenerated, notebook trimmed) |
| `.DS_Store`, `.idea/`, `__pycache__` present in tree | `.gitignore` covers them | Fixed (kept ignored) |
| `saved-face-embeddings/` not tracked | `git status: ??` | Fixed (committed; 240 KB, small enough) |
| No branches/tags/releases | single `main`, no tags | Fixed (tag the paper build) |

Note: the repository has a GitHub remote (`NicGroenewald/Face-Embedding-Map`). Nothing in this build
is pushed without explicit approval.

---

## 6. What this build changes (map to tasks)

| task | owner | addresses |
|---|---|---|
| T1 `fem` library | core-dev | 2.1, 2.5, 1.6 |
| T2 refactor + tests + config | core-dev | 2.1–2.5, 4.3, 5 |
| T3 Olivetti E1/E2 + model arch | lab | 1.1, 1.7, 1.8, 1.9 |
| T4 live lighting study | lab | 1.3–1.5, 4.2 |
| T5 threshold analysis | lab | 1.2 (partly), 1.7 |
| T6 findings + notebook rewrite | lab | 3.3, 3.4, 1.3–1.5 |
| T7 paper skeleton + diagrams | paper | 3.6 |
| T8 paper results + compile | paper | 3.6, 1.10 |
| T9 README + integration + commits | lead | 3.1, 3.2, 4.1, 5 |

## 7. Backlog after this build

1. **Stranger captures** under the same six lighting conditions (the missing negative class).
2. **Yale B** (Stage 6) for a controlled illumination-direction experiment with real faces, not one
   subject under a phone flash.
3. **LFW** for an unconstrained benchmark and a cross-dataset test of the same frozen model.
4. **Triplet-loss fine-tuning** (Stage 7) — the only stage where "training" is meaningful; keep the
   no-training rule until then.
5. **Matched-dimension deep features**: e.g. also evaluate Map B's 512-d vectors after PCA to the
   same 2-D and full-dim protocol, and compare nearest-neighbour confusion sets directly (partly
   done in E2's confusion analysis).
6. **Cross-validation over identity splits** for the verification protocol (currently 5 random
   5/5 splits of the same 40 people).
7. **CI** (`pytest` + paper build) and a license.
8. **Live capture metadata**: record camera distance, phone position, ambient lux, and a
   per-frame quality score.

## 8. Things that are already right (do not break them)

- The "no training before Stage 7" rule — the comparison is only meaningful while the model is
  frozen and the evaluation is honest.
- Measuring in full dimensionality and plotting in 2-D — the project got this right before writing
  most of the code; it is the reason the numbers can be trusted.
- Using PCA rather than t-SNE for anything reusable — a genuine methodological insight, and the
  "t-SNE is a plotter, not an embedder" note is worth keeping in the paper's discussion.
- Euclidean-on-unit-vectors is equivalent to cosine ranking; the findings' choice of Euclidean is
  fine, it just needed the conversion stated.
- Keeping the MTCNN/embedder device split (CPU detector, MPS embedder) — that is a real Mac-specific
  finding and belongs in the paper.
- The findings format (Observed / Means / So what) and the honest "confounded earlier test"
  section — keep both in the rewritten docs.
