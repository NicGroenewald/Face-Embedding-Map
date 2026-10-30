# Code Map — how the pipeline fits together

*Owner: core-dev. This is the "where does this live?" document for the library
and scripts. The build contract (experiments, JSON schemas, paper spec) is
`docs/DESIGN.md`; this file explains the implementation.*

---

## The one rule: don't build the pipeline twice

`docs/PLAN.md` trap #4 says it plainly: *"Load → embed → squash → plot
→ score. Only the embed step swaps. If I'm copy-pasting plotting code, stop and
pull it into a function."* The codebase is organised around that:

| Stage | Only implementation |
|---|---|
| load Olivetti | `fem.data.load_olivetti` |
| load live scans + conditions | `fem.data.load_scans` (manifest-driven) |
| preprocess Olivetti for the CNN | `fem.preprocess.to_model_input` |
| embed faces | `fem.embed.FaceEmbedder` |
| cache Olivetti embeddings | `fem.embed.embed_olivetti` |
| PCA / eigenfaces / 2-D fidelity | `fem.maps` |
| silhouette / k-NN / ROC / EER / thresholds | `fem.metrics` |
| every result figure (PNG + PDF) | `fem.plotting` |

Map A (4096-d raw pixels) and Map B (512-d embeddings) differ **only** in the
matrix handed to `fem.maps` / `fem.metrics`. If a runner is about to write its
own ROC curve, distance loop, or `plt.savefig`, it should call `fem` instead.

---

## Files

### `fem/` — the library

| module | responsibility | why it exists |
|---|---|---|
| `config.py` | repo paths, seed, sizes, `ensure_dirs()` | one place that knows where data/results/figures live; paths derived from `__file__` so cwd does not matter |
| `data.py` | Olivetti loader; `Scan`/`ScanSet`; manifest-driven live-scan loader | the `.npy` filenames encode a timestamp, not a lighting condition; `data/scans.json` is the source of truth, so condition metadata can be fixed without touching the data |
| `preprocess.py` | `to_model_input`, `standardize` | Olivetti must be fed the same scale MTCNN uses: greyscale → RGB, bilinear resize 64→160 (`antialias=True`), ×255, `(x − 127.5) / 128`. `fixed_image_standardization` is imported from facenet-pytorch rather than copied |
| `embed.py` | frozen InceptionResnetV1(vggface2) on MPS/CPU; cached Olivetti embeddings | one model instance, `eval()` + `no_grad`, output verified unit-norm; the 400-image embedding is cached to `results/embeddings_olivetti.npz` |
| `maps.py` | `pca_map`, `explained_variance`, `variance_milestones`, `eigenfaces`, `rank_correlation_2d` | one PCA recipe for both maps (`svd_solver="randomized"`, `random_state=SEED`, no StandardScaler); the 2-D-vs-full Spearman number quantifies how much the scatter plot misleads |
| `metrics.py` | distances, silhouette, k-NN LOO, pair split, ROC-AUC, EER, TAR@FAR, threshold sweeps, summary | every verification number in the paper comes from here; conventions are documented in the module docstring |
| `plotting.py` | all figure savers | identical style/palette/size for Map A and Map B; each saver writes **PNG (200 dpi) + PDF** from one path |
| `__init__.py` | package surface | `import fem` is cheap: numpy/sklearn modules eager, torch/matplotlib modules lazy (`fem.embed`, `fem.preprocess`, `fem.plotting`) |

### Runners (owned by `lab`)

| script | produces |
|---|---|
| `analysis/run_olivetti.py` | E1+E2 → `results/olivetti.json`, `results/embeddings_olivetti.npz`, figures |
| `analysis/run_live.py` | E3 → `results/live.json`, figures |
| `analysis/run_threshold.py` | E4 → `results/threshold.json`, figure |
| `analysis/model_arch.py` | E5 → `results/model_arch.json` |
| `analysis/_common.py` | results JSON header/version block, NumPy-scalar-safe `save_json` |

Runners orchestrate; they must not re-implement `fem`.

### Data-capture and sanity scripts

| script | role |
|---|---|
| `scripts/live_capture.py` | webcam capture CLI: MTCNN on CPU, embedder on MPS, warm-up frames discarded, one detection per frame, unique metadata sidecar, private gitignored output by default |
| `scripts/compare_scans.py` | human-readable distance report over the saved scans (jitter, repeat scans, intensity/direction effects, cosine equivalents) with PASS/FAIL self-checks and non-zero exit on failure |

---

## Data and artifacts

```
data/sklearn/                 ignored cache: fetch_olivetti_faces(data_home=...)
data/scans.json               tracked manifest for the 12 live scans (source of truth)
data/scans/*.npy             tracked historical biometric templates (10, 512), unit norms
data/private-scans/           gitignored default output for new biometric captures
results/*.json                tracked experiment outputs (written by analysis/)
results/embeddings_olivetti.npz   ignored cache (deterministic, regenerable)
figures/*.png + *.pdf         tracked result figures (written by analysis/)
```

`load_scans()` validates that every row is unit norm. That is not paranoia:
`InceptionResnetV1.forward` ends in `F.normalize`, so all Euclidean↔cosine
conversions (`cos = 1 − d²/2`) assume unit vectors, and a violated invariant
must fail loudly rather than produce plausible-looking wrong numbers.

### Capturing additional private scans

1. Capture: `python scripts/live_capture.py --intensity flash --direction right --repeat 3`
   (writes a unique timestamped `.npy` and `.json` under gitignored `data/private-scans/`).
2. Keep biometric templates private and get informed subject consent before publication.
   The sidecar's unique `name` is a capture ID; its `label` is human-readable metadata.
3. **Do not append an entry to `data/scans.json` and assume `make live` will include it.**
   `analysis/run_live.py` intentionally asserts the historical twelve scans and two repeats per
   condition. A new study needs a subject-aware manifest and runner (see `docs/ROADMAP.md`).

---

## Conventions and invariants (the things that silently break results)

* **Distances are Euclidean.** A verifier accepts when `distance <= threshold`.
  FAR is the fraction of impostor trials accepted, FRR the fraction of genuine
  trials rejected. EER interpolates between discrete thresholds; it need not
  be the empirical FAR/FRR at one attainable threshold.
* **`cosine_from_euclidean` returns cosine similarity**, not distance:
  `1 − d²/2` (0 = orthogonal, 1 = identical).
* **Measure in full dimensionality.** `silhouette` and `knn_loo` take the
  4096-d / 512-d matrices; the 2-D PCA coordinates are for plots only.
* **Same PCA settings everywhere.** `pca_map` is the only PCA; identical seeds
  for both maps. No `StandardScaler` (documented rejection: it amplifies
  near-constant background pixels).
* **One detection per frame.** In `scripts/live_capture.py` the box from `mtcnn.detect()` is
  reused for both the HUD rectangle and `mtcnn.extract()`; `mtcnn(rgb)` would
  detect a second time.
* **Device split.** MTCNN runs on CPU (it cannot run on MPS on this machine);
  the embedder runs on MPS when available. `FaceEmbedder(device="mps")` raises
  rather than silently falling back to CPU, so results JSON cannot lie about
  the device.
* **Seed = 0** for every random choice (`fem.config.SEED`).
* **No NumPy scalars in JSON.** `analysis/_common.py:to_jsonable` handles it;
  run through it rather than hand-casting.

---

## Running things

Activate an environment installed from `requirements.txt` (see `README.md`); the Makefile uses
`python3` from PATH by default and sets a writable matplotlib cache. Override with
`make PY=/path/to/python all` when needed. An installed wheel can point at a source checkout via
`FEM_PROJECT_ROOT=/path/to/checkout` (dataset and generated results are not bundled).

`make` targets:

```sh
make test           # pytest, offline, no webcam, no model download
make maps           # E1+E2 -> results/olivetti.json + figures
make live           # E3    -> results/live.json + figures
make threshold      # E4    -> results/threshold.json
make arch           # E5    -> results/model_arch.json
make compare        # pretty distance report for the saved scans
make paper          # pdflatex twice
make all            # test + all experiments
```

Direct equivalents (from an activated environment, at repo root):

```sh
export MPLCONFIGDIR="$PWD/.cache/mpl"
python -m pytest -q
python analysis/run_olivetti.py
python scripts/compare_scans.py
python scripts/live_capture.py --help
```

---

## Tests

`tests/` is offline and runs in under ten seconds.  It covers the parts where
a silent bug would corrupt the paper:

| file | covers |
|---|---|
| `test_metrics.py` | hand-computed distances, cosine relation, silhouette, k-NN LOO, pair split, ROC-AUC, EER, conservative TAR, threshold sweep, summary, d′ |
| `test_maps.py` | PCA recovers a synthetic rank-2 subspace exactly, eigenface reshaping, variance milestones, 2-D rank correlation |
| `test_data.py` | Olivetti shapes/labels, manifest integrity (12 scans × 10 frames, condition counts, unit norms), missing-file/unit-norm errors |
| `test_preprocess.py` | exact `(x − 127.5)/128` values for 0/0.5/1, greyscale→RGB replication, shape/range validation |
| `test_plotting.py` | every saver writes PNG at 200 dpi **and** a valid PDF; constant-series confusion panel does not emit NaN |
| `test_live_capture.py` | argument parsing, label sanitising, largest-box choice, warm-up discarding, sidecar contents, no-overwrite behaviour |
| `test_compare_scans.py` | self-check semantics, report grouping, non-zero exit on missing data |

`tests/conftest.py` sets `MPLCONFIGDIR`, the Agg backend and `sys.path` before
anything imports matplotlib.

---

## Where to look first when something is wrong

1. `import fem` fails → a module-level import error (torch/matplotlib are lazy;
   `fem.data`/`fem.maps`/`fem.metrics` are not).
2. `load_scans` raises → manifest/file disagreement or non-unit rows; the error
   names the scan.
3. Numbers look impossible → check the metric convention in
   `fem/metrics.py`'s module docstring before suspecting the data.
4. Figures missing a PDF → call the saver with the base path and let `_save`
   write both; do not call `plt.savefig` directly in a runner.
5. Embedding differs from the live pipeline → confirm `to_model_input` is the
   shortcut being used (no detector for pre-aligned Olivetti; same
   standardisation otherwise).
