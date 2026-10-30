# Face Embedding Map — Build Contract & Experiment Plan

*Owner: Lead. Every teammate reads this before writing anything. If something here is wrong or
impossible, message the Lead instead of silently diverging.*

> **Historical note.** This document was the build contract written *before* the October 2026
> rebuild. Some statements below describe the starting state ("Map B: not built yet"), which the
> build then resolved. The original specification overstated open-set evaluation and implied
> cross-domain distances bounded stranger scores; those claims were corrected in the final paper.
> For the finished state read `README.md`, `docs/CODE.md`, `results/README.md` and
> `docs/ROADMAP.md`; §3 has been updated to the final layout.

This was the design contract for:

1. what the project is trying to prove,
2. the environment to run things in,
3. the `fem` library API (the contract between code and experiments),
4. the exact experiments, metrics and protocols,
5. the `results/*.json` schema the paper consumes,
6. the figures the paper consumes,
7. file ownership (write scopes) so nobody overwrites anybody,
8. conventions (seeds, paths, formatting).

---

## 1. The research question

> **Do learned face embeddings separate *identity* better than raw pixels do — and can we show it
> with a number, not just a nice-looking picture?**

The project already contains:

- **Map A**: Eigenfaces — PCA on the 4096 raw pixel values of the Olivetti faces (400 photos,
  40 people, 10 each). Notebook + findings exist (`notebooks/`).
- **Map B**: not built yet. InceptionResnetV1 pretrained on VGGFace2, frozen, evaluated on the same
  Olivetti faces → 512-d unit vectors, same PCA settings, same plots.
- **Stage 5 live study**: 12 webcam scans of one person (the author) under six lighting conditions,
  10 frames each, saved as `data/scans/*.npy`. Existing findings document
  (`docs/findings/live-illumination.md`).

What is missing, and what this build adds:

- **Stage 3**: Map B on Olivetti (never run).
- **Stage 4**: quantitative comparison in *full dimensionality* — silhouette, 1-nearest-neighbour
  leave-one-out, genuine/impostor ROC-AUC and EER — Map A vs Map B.
- **Stage 5b**: known-population cross-identity verification on Olivetti, with the Map A vs Map B
  comparison; the original "open-set FaceID" wording was too broad without independent strangers.
- A cleaned-up, reproducible codebase and test suite.
- A research paper (LaTeX → PDF) with architecture diagrams and result figures.

Everything above is **frozen-model, no training**. There is no train/test split and no learning
anywhere. Stage 7 (triplet-loss training) stays out of scope and is future work in the paper.

---

## 2. Environment (read this twice)

Use the existing project virtualenv — **do not** create a new one, do not pip install into it:

```
/Users/nic/.venvs/nn-crew/bin/python
```

It contains exactly the versions in `requirements.txt` (numpy 2.5.3, scikit-learn 1.9.1,
matplotlib 3.11.2, torch 2.14.0, facenet-pytorch 2.5.3, opencv-python 5.0.0.93), plus pytest and
scipy.

**Always** export a writable matplotlib cache before running anything (the home-directory cache is
not writable in this sandbox):

```sh
export MPLCONFIGDIR="$PWD/.cache/mpl"
```

`make` targets do this for you.

**Already downloaded / cached (do not re-download):**

- Olivetti faces: `data/sklearn/` (fetched with `fetch_olivetti_faces(data_home="data/sklearn")`).
- VGGFace2 weights: `~/.cache/torch/checkpoints/20180402-114759-vggface2.pt` (read-only, works).
- Live embeddings: `data/scans/*.npy`, shape `(10, 512)` float32, **unit norm already**
  (facenet-pytorch L2-normalises its output).
- MPS is available on this Mac. `torch.backends.mps.is_available() == True`.

**Network** is available (Olivetti download, arXiv lookups). Do not write outside the repo except
for read-only caches.

**Do not install packages, do not upgrade anything.** If something seems missing, tell the Lead.

### Reproducibility rules

- `SEED = 0` everywhere; pass `random_state=SEED` to every sklearn estimator.
- The CNN is deterministic in `eval()` mode; embeddings are cached to `results/embeddings_*.npz` by
  the experiment runner so later steps don't depend on re-running the model.
- Every number that reaches the paper comes from a `results/*.json` file produced by a script in
  `analysis/`. **No hand-typed numbers anywhere.**
- Record versions/hardware/date in every results JSON (see schema).

---

## 3. Repository layout (final)

```
.
├── README.md                     # front door: question, headline numbers, how to run, what I got wrong
├── Makefile                      # make test | maps | live | threshold | arch | compare | diagrams | paper
├── pyproject.toml                # package metadata (fem, scripts) + pytest config
├── requirements.txt              # pinned runtime + test dependencies
├── .gitignore / .gitattributes
├── fem/                          # the library — one module per pipeline stage
│   ├── config.py                 # paths, seed, constants
│   ├── data.py                   # Olivetti loader; manifest-driven ScanSet loader
│   ├── preprocess.py             # (x-127.5)/128 recipe shared with the MTCNN path
│   ├── embed.py                  # frozen InceptionResnetV1 + cached embeddings
│   ├── maps.py                   # PCA maps, eigenfaces, 2-D-vs-full rank correlation
│   ├── metrics.py                # silhouette, k-NN, ROC, EER, TAR@FAR, threshold sweeps
│   └── plotting.py               # every result figure (PNG 200 dpi + vector PDF)
├── analysis/                     # experiment runners; each writes results/*.json + figures/
│   ├── run_olivetti.py           # E1 + E2  (Map A vs Map B)
│   ├── run_live.py               # E3       (webcam illumination study)
│   ├── run_threshold.py          # E4       (known-population verification)
│   ├── model_arch.py             # E5       (verified layer shapes)
│   └── _common.py                # JSON header/provenance + numpy-safe serialisation
├── scripts/                      # standalone command-line tools
│   ├── live_capture.py           # webcam capture CLI (one detect/frame, metadata, never overwrites)
│   └── compare_scans.py          # human-readable scan-distance report
├── data/
│   ├── scans.json                # scanned manifest: scan name -> file + condition metadata
│   ├── scans/                    # 12 raw webcam scans, (10, 512) unit-norm float32
│   ├── README.md                 # what is tracked vs cached
│   └── sklearn/                  # IGNORED cache: fetch_olivetti_faces(data_home=...)
├── results/
│   ├── *.json                    # machine-readable results (source of truth for the paper)
│   ├── *.npz                     # IGNORED deterministic caches (embeddings, map coords)
│   └── README.md                 # schema + how to regenerate
├── figures/
│   ├── *.png / *.pdf             # result figures, generated from results
│   └── README.md                 # name → producing script
├── notebooks/
│   └── 01-map-a-raw-pixels.ipynb # cleaned exploration notebook (drives fem; no duplicated pipeline)
├── docs/
│   ├── DESIGN.md                 # this build contract
│   ├── REVIEW.md                 # audit of the original repository
│   ├── CODE.md                   # implementation map
│   ├── ROADMAP.md                # prioritized next steps
│   ├── PLAN.md                   # the original project plan + traps list
│   └── findings/
│       ├── live-illumination.md  # Stage 5 findings (Observed / Means / So what)
│       └── map-a-raw-pixels.md   # Map A findings
├── paper/
│   ├── main.tex / main.pdf       # compiled research paper
│   ├── sections/*.tex            # one file per section
│   ├── refs.tex                  # verified bibliography
│   └── figures/
│       ├── fig_*.pdf             # architecture diagrams (TikZ)
│       └── src/                  # diagram sources (style.tex + fig_*.tex)
└── tests/                        # pytest; fast, offline, headless
    ├── conftest.py
    ├── test_{metrics,maps,data,preprocess,plotting}.py
    └── test_{live_capture,compare_scans}.py
```

Rules that keep it that way:

- Root holds only the files a newcomer needs: README, build config, `.gitignore`, `.gitattributes`.
  No scripts, no data, no findings at the root.
- Exactly one implementation per pipeline stage, in `fem/`. Runners orchestrate, they never
  re-implement. If a runner is about to write its own ROC curve or `plt.savefig`, it should call
  `fem` instead.
- `results/*.json` are source-of-truth deliverables; `results/*.npz`, `data/sklearn/` and `.cache/`
  are regenerable caches and are ignored.
- Paths are derived from `fem.config` (`ROOT` from `__file__`), never from the current working
  directory, so `python analysis/run_live.py` works from anywhere.

## 4. `fem` API contract

These signatures are fixed — experiments and tests are written against them. Implement them exactly;
add private helpers freely. Type hints + short docstrings. Pure functions where possible.

```python
# fem/config.py
from pathlib import Path
ROOT: Path            # repo root, computed from __file__ (works when cwd != root)
DATA_DIR: Path        # ROOT / "data"
RESULTS_DIR: Path     # ROOT / "results"
FIGURES_DIR: Path     # ROOT / "figures"
SCANS_DIR: Path       # ROOT / "data/scans"
PAPER_DIR: Path       # ROOT / "paper"
SEED: int = 0
FACE_SIZE: int = 160          # model input resolution
EMBED_DIM: int = 512
IDENTITIES: int = 40          # Olivetti
IMAGES_PER_IDENTITY: int = 10
LIVE_SCANS: int = 12
FRAMES_PER_SCAN: int = 10
def ensure_dirs() -> None    # mkdir -p results, figures
```

```python
# fem/data.py
def load_olivetti(data_home: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (images, labels): images float32 (400, 64, 64) in [0, 1], labels int (400,) in 0..39."""

@dataclass(frozen=True)
class Scan:
    name: str               # e.g. "room_front_1"
    path: Path
    intensity: str          # "room" | "flash"
    direction: str          # "front" | "left" | "right" | "bottom"
    frames: np.ndarray      # (10, 512) float32 unit-norm rows

@dataclass
class ScanSet:
    scans: list[Scan]
    def stack(self) -> np.ndarray:      # (12, 10, 512)
    def names(self) -> list[str]
    def frames(self) -> np.ndarray:     # (120, 512) all frames in scan order
    def mean_embeddings(self) -> np.ndarray   # (12, 512), re-normalised scan means
    def labels(self) -> np.ndarray      # (12,) condition strings like "flash/left"
def load_scans(manifest: str | Path | None = None) -> ScanSet
    """Load from data/scans.json. Manifest is the source of truth for condition metadata."""
```

`data/scans.json` format (core-dev writes it from the existing filenames):

```json
{
  "format_version": 1,
  "subject": "nic",
  "captured": "2026-09-06",
  "frames_per_scan": 10,
  "scans": [
    {"name": "room_front_1", "file": "06092026-light_face_front_1.npy",
     "intensity": "room", "direction": "front", "repeat": 1},
    {"name": "flash_left_2", "file": "06092026-phone_light_face_left_2.npy",
     "intensity": "flash", "direction": "left", "repeat": 2}
  ]
}
```

```python
# fem/preprocess.py
def to_model_input(images: np.ndarray, size: int = FACE_SIZE) -> "torch.Tensor":
    """(N, 64, 64) float in [0,1] (or (N,3,64,64)) -> (N, 3, size, size) float32.

    Recipe (must match the live MTCNN path exactly):
      1. replicate greyscale to 3 channels,
      2. bilinear resize to (size, size), antialias=True,
      3. scale to 0..255 then fixed_image_standardization: (x - 127.5) / 128.
    """
def standardize(batch_uint8: "torch.Tensor") -> "torch.Tensor":
    """(x - 127.5) / 128 on 0..255 input. Use facenet_pytorch.models.mtcnn.fixed_image_standardization."""
```

```python
# fem/embed.py
class FaceEmbedder:
    """Frozen InceptionResnetV1(vggface2). Output rows are L2-normalised (512-d)."""
    def __init__(self, device: str | None = None):  # device: "mps" if available else "cpu"
    def embed_images(self, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """(N,64,64) float -> (N,512) float32 unit-norm. torch.no_grad, model.eval().
        Prints/records batch progress only when verbose."""
    def embed_tensor(self, x: "torch.Tensor") -> np.ndarray:  # already model-input
def embed_olivetti(embedder: FaceEmbedder | None = None, cache: Path | None = None) -> np.ndarray:
    """Cached: returns (400,512); writes/reads results/embeddings_olivetti.npz keyed by seed/model."""
```

```python
# fem/maps.py
@dataclass
class PcaMap:
    model: "sklearn.decomposition.PCA"   # fitted
    coords: np.ndarray                    # (N, 2)
    evr: np.ndarray                       # explained_variance_ratio_, length n_components
def pca_map(X: np.ndarray, n_components: int = 2, seed: int = SEED) -> PcaMap
    """PCA(n_components, svd_solver='randomized', random_state=seed), centred (sklearn default).
    NOTE: no StandardScaler — documented as a rejected choice (see Map A findings §4)."""
def explained_variance(X: np.ndarray, n_components: int = 100, seed: int = SEED) -> np.ndarray
def variance_milestones(evr: np.ndarray, milestones=(25, 50, 75, 90, 95)) -> dict[int, int]
    """{milestone_percent: n_components_needed}"""
def eigenfaces(components: np.ndarray, n: int = 16, image_shape=(64, 64)) -> np.ndarray
def rank_correlation_2d(coords: np.ndarray, X_full: np.ndarray, n_pairs: int = 20000,
                        seed: int = SEED) -> float
    """Spearman correlation between 2-D and full-dimensional pairwise Euclidean distances.
    Quantifies how much of the true geometry the 2-D plot preserves."""
```

```python
# fem/metrics.py
def pairwise_distances(A: np.ndarray, B: np.ndarray | None = None,
                       metric: str = "euclidean") -> np.ndarray
def cosine_from_euclidean(d: np.ndarray) -> np.ndarray
    """For unit vectors: cos = 1 - d^2 / 2."""
def silhouette(X: np.ndarray, labels: np.ndarray) -> float
    """sklearn.metrics.silhouette_score(X, labels, metric='euclidean') on FULL-dimensional X."""
def knn_loo(X: np.ndarray, labels: np.ndarray, k: int = 1) -> dict
    """Leave-one-out k-NN, self excluded. Returns {'accuracy': float, 'top5': float,
    'predictions': int[], 'errors': int[]}."""
def pair_distances(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]
    """All unordered pairs -> (genuine_distances, impostor_distances)."""
def roc_auc(genuine: np.ndarray, impostor: np.ndarray) -> float
def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]
    """Returns (equal_error_rate, threshold). Interpolate FAR=FRR on a fine threshold grid."""
def tar_at_far(genuine: np.ndarray, impostor: np.ndarray, far: float) -> float
def threshold_sweep(genuine: np.ndarray, impostor: np.ndarray,
                    thresholds: np.ndarray | None = None) -> dict
    """dict with 'thresholds', 'far', 'frr' arrays for FAR/FRR-vs-threshold plots."""
def summary(genuine: np.ndarray, impostor: np.ndarray) -> dict
    """mean/std/min/max for each class + d_prime, auc, eer, eer_threshold,
    tar_at_far_1e-3, tar_at_far_1e-2."""
```

```python
# fem/plotting.py     (every saver writes BOTH .png at 200 dpi and .pdf, vector)
def save_scatter(coords, labels, path, title="", identity_names=None, cmap=None) -> None
def save_eigenfaces(components, path, n=16, image_shape=(64, 64)) -> None
def save_scree(evr, path, title="") -> None
def save_distance_histograms(data: dict, path, xlabel="Euclidean distance") -> None
    # data: {label: (genuine_array, impostor_array)}
def save_roc(curves: dict, path) -> None      # curves: {label: (genuine, impostor)}
def save_threshold_sweep(curves: dict, path) -> None
def save_heatmap(matrix, tick_labels, path, title="", vmin=None, vmax=None) -> None
def save_metric_bars(metrics: dict, path, ylabel="") -> None   # grouped bars, Map A vs Map B
def save_lighting_panel(live: dict, path) -> None              # jitter vs direction effect panel
```

`plotting` must not hardcode figure aesthetics per call site: one `_save(fig, path)` helper
(png+pdf), tight layout, readable fonts, colorblind-safe palette, identity colours consistent
between Map A and Map B scatter plots.

```python
# fem/pipeline.py  (optional convenience; analysis scripts may use or bypass)
def run_olivetti(verbose=True) -> dict      # returns the results dict, writes results/olivetti.json
def run_live(verbose=True) -> dict
def run_threshold(verbose=True) -> dict
```

---

## 5. Experiments (exact protocol)

All distances are Euclidean unless stated. Embeddings are unit vectors, so report the cosine
equivalent where it aids interpretation (`cos = 1 - d²/2`).

### E1 — Map A: Eigenfaces on Olivetti

1. Load 400×64×64; flatten to (400, 4096).
2. `pca_map(X, n_components=2, seed=0)` → 2-D scatter coloured by identity.
3. `explained_variance(X, n_components=100)` → scree + cumulative plot; milestones dict.
4. Eigenfaces: first 16 components reshaped to 64×64 (grid).
5. **Full-dimensional metrics on the 4096-d vectors:**
   - silhouette (40 labels),
   - 1-NN LOO accuracy + top-5,
   - genuine/impostor pair distances (1800 genuine, 78 000 impostor),
   - ROC-AUC, EER + threshold, TAR@FAR=1e-3, TAR@FAR=1e-2, d′.
6. 2-D geometry fidelity: `rank_correlation_2d`.
7. Seed robustness: repeat step 2 with seeds 0, 1, 2 and report the three Spearman values and the
   pairwise rank correlation of the 2-D embeddings across seeds. Also state explicitly that the
   full-dimensional metrics are seed-independent (deterministic PCA/SVD for silhouette).
8. Save a `.npz` with coords, labels, evr, components for figure regeneration.

### E2 — Map B: frozen InceptionResnetV1 on Olivetti

1. Preprocess with `to_model_input` (bilinear resize 64→160, greyscale→RGB, (x−127.5)/128).
   **This normalisation is mandatory and must match the MTCNN path the live embeddings used.**
2. Embed 400 images → (400, 512), unit norm. Cache to `results/embeddings_olivetti.npz`.
3. Same PCA settings as E1 (`n_components=2`, seed 0), same colours, same plot style.
4. Explained variance milestones for 100 components (512-d space).
5. Same full-dimensional metrics on the 512-d vectors as E1 step 5.
6. `rank_correlation_2d`.
7. Seed robustness as E1 step 7.
8. **Confusion analysis (answers "are the same people confused in both maps?"):**
   - per-identity 1-NN LOO accuracy for Map A and Map B; Pearson correlation across the 40 values,
   - for each image compare the predicted identity under Map A and Map B: count same/different,
   - where both maps err, do they choose the same wrong identity? Report overlap fraction.
9. Save a `.npz` as E1.

### E3 — Live lighting study (existing 12 scans × 10 frames)

Design is unbalanced by necessity: room light has front + left only; flash has front, left, right,
bottom; two scans (repeat 1 and 2) per condition; one subject.

1. **Per-scan noise floor:** all 45 frame pairs within each scan: min/mean/max. Also report the
   mean over scans.
2. **Same-condition between-scan:** 100 frame pairs for the two repeats of each condition.
3. **Intensity effect (front only):** `room_front_*` vs `flash_front_*` frame-pair distances.
4. **Direction effect (flash only):** all six direction pairs (front-left, front-right, front-bottom,
   left-right, left-bottom, right-bottom): min/mean/max of frame-pair distances.
5. **Ratio:** `mean(direction effect) / mean(same-condition noise floor)` — the headline "lighting
   direction costs N× frame jitter".
6. **Pooling effect:** compare 12×12 scan-mean distances against the frame-level distances for the
   same scan pair. Report mean reduction (%) — support for "average N frames at enrolment".
7. **12×12 heatmap** of mean frame-pair distance between scans, ordered room-front, flash-front,
   flash-left, flash-right, flash-bottom.
8. **Observed single-reference contrast:** worst genuine frame-pair distance over all 12 scans
   (max), and the same after frame averaging. A lower threshold rejects *that pair* with one
   reference; it is not a bound for a five-reference verifier.
9. **Cross-domain sanity check:** distance from each live scan to the 400 Olivetti embeddings
   (E2, same model). Label it explicitly as domain-confounded (different capture, crop and
   resolution): neither upper nor lower bound on webcam strangers, not an impostor distribution.
   Include it in JSON with a `caveat` string the paper must repeat.
10. Euclidean↔cosine conversion for every headline pair.

### E4 — Known-population verification (controlled, on Olivetti)

This is the Map A vs Map B comparison the plan calls the potential headline result.

Protocol (run identically for Map A pixels and Map B embeddings):
- For each identity `p` in 0..39: enrol 5 images, probe the other 5 (genuine); impostor probes are
  all 10 images of the other 39 identities.
- Split selection: **(a) primary** = 5 random 5/5 splits (seeded 0–4) averaged, and **(b)** secondary
  = fixed first-half/second-half split (reported as a session-bias check in the JSON, not the paper
  headline).
- Score = distance; decision = accept if distance ≤ threshold.
- Per split: FAR/FRR over a threshold grid, EER, TAR@FAR=1e-3, TAR@FAR=1e-2, AUC over all trials.
- Aggregate: mean ± std across splits; store the operating point at the EER threshold and at a
  "convenience" threshold that gives FRR = 1 % (report corresponding FAR).
- Question to answer in one sentence in the JSON: *at the same false-accept rate, how much better is
  Map B than Map A?*

### E5 — Model architecture dump (feeds the paper diagram only)

Hook every top-level layer of InceptionResnetV1, run one 160×160 input, record
`(name, module type, output shape, param count)`. Save `results/model_arch.json` plus the total
parameter count and the number of multiplications as reported by hooks? (FLOPs not required — do not
invent them.) The paper diagram must use these verified shapes, not guessed ones.

### Optional E6 — synthetic lighting direction (only if time allows)

Apply a smooth horizontal/vertical illumination gradient to Olivetti images (same identity, same
pose, lighting only), re-embed, and measure identity-preserving embedding shift; compare with Map A.
Explicitly synthetic; if run, label it as an ablation. Do **not** let it delay E1–E5.

---

## 6. `results/*.json` schema

Common header for every file:

```json
{
  "schema_version": 1,
  "created": "2026-10-30T10:00:00Z",
  "git_commit": "abc1234",
  "python": "3.12.14",
  "packages": {"numpy": "2.5.3", "scikit-learn": "1.9.1", "torch": "2.14.0",
               "facenet-pytorch": "2.5.3", "matplotlib": "3.11.2"},
  "device": "mps",
  "seed": 0,
  "notes": []
}
```

`results/olivetti.json`:

```json
{
  "common": {...},
  "dataset": {"name": "Olivetti", "n_images": 400, "n_identities": 40,
              "images_per_identity": 10, "image_shape": [64, 64], "pixel_range": [0.0826, 0.8760],
              "aligned": "pre-aligned, tightly cropped (documented limitation)"},
  "map_a": {
    "space": "raw pixels (4096-d)",
    "variance_milestones": {"25": 2, "50": 4, "75": 19, "90": 66, "95": null},
    "component0_evr": 0.23813,
    "pc1_interpretation": "illumination gradient (left-right brightness)",
    "silhouette": 0.0,
    "knn_loo": {"accuracy": 0.0, "top5": 0.0},
    "pairs": {"genuine": {"n": 1800, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
              "impostor": {"n": 78000, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}},
    "roc": {"auc": 0.0, "eer": 0.0, "eer_threshold": 0.0,
            "tar_at_far_1e-3": 0.0, "tar_at_far_1e-2": 0.0, "d_prime": 0.0},
    "rank_correlation_2d": 0.0,
    "seed_robustness": {"seeds": [0,1,2], "spearman_2d_vs_full": [0,0,0],
                        "pairwise_2d_rank_corr": [[1,0,0],[0,1,0],[0,0,1]]},
    "per_identity_knn_accuracy": [40 numbers]
  },
  "map_b": { ...same keys, "space": "InceptionResnetV1 vggface2 (512-d unit)" ... },
  "confusion": {"per_identity_accuracy_corr_pearson": 0.0,
                "same_prediction_fraction": 0.0,
                "both_wrong_same_choice_fraction": 0.0}
}
```

`results/live.json`:

```json
{
  "common": {...},
  "design": {"subject": "nic", "scans": 12, "frames_per_scan": 10,
             "captured": "2026-09-06", "conditions": ["room/front","room/left",
             "flash/front","flash/left","flash/right","flash/bottom"],
             "imbalance_note": "room light has front+left only"},
  "within_scan_jitter": [{"scan": "room_front_1", "min": 0.0, "mean": 0.0, "max": 0.0}, ...],
  "same_condition": [{"condition": "room/front", "min": 0.0, "mean": 0.0, "max": 0.0}, ...],
  "noise_floor_mean": 0.0,
  "intensity_effect": {"condition_pair": ["room/front", "flash/front"], "mean": 0.0,
                       "min": 0.0, "max": 0.0, "note": "indistinguishable from noise"},
  "direction_effect": {"pairs": [{"pair": ["flash/front","flash/left"], "mean": 0.0,
                       "min": 0.0, "max": 0.0}, ...], "mean": 0.0},
  "direction_to_noise_ratio": 0.0,
  "pooling": {"frame_level_mean": 0.0, "scan_mean_level_mean": 0.0, "reduction_pct": 0.0},
  "worst_genuine_distance": {"frame_level": 0.0, "scan_mean_level": 0.0,
                             "pair": ["flash/right_2", "flash/bottom_2"],
                             "cosine_equivalent": 0.0},
  "cross_domain_impostor": {"olivetti_min_distance_to_any_live_scan": 0.0,
                            "olivetti_mean_of_min": 0.0,
                            "caveat": "different capture/crop/resolution; no bound on webcam strangers"},
  "matrix_names": ["room_front_1", ...],
  "matrix_mean_distance": [[12x12]]
}
```

`results/threshold.json`:

```json
{
  "common": {...},
  "protocol": {"enrol": 5, "probe": 5, "impostor_probes_per_identity": 390,
               "splits": "5 random 5/5 splits, seeds 0-4", "identical_for": ["map_a","map_b"]},
  "map_a": {"per_split": [{"seed":0,"eer":0.0,"eer_threshold":0.0,"tar_at_far_1e-3":0.0,
             "tar_at_far_1e-2":0.0,"auc":0.0,"far_at_frr_1pct":0.0}],
            "mean": {...}, "std": {...}},
  "map_b": {...},
  "headline": {"statement": "...", "map_b_eer": 0.0, "map_a_eer": 0.0,
               "far_gap_at_matched_frr": "..."},
  "session_bias_check": {"map_a": {...}, "map_b": {...}, "note": "fixed split, not primary"}
}
```

Rules for the JSON:
- `null` is allowed only where a thing genuinely does not exist (e.g. 95 % milestone not reached in
  the first 100 components) — say why in `notes`.
- Every table/figure in the paper must be traceable to a key here. The paper writer should be able
  to `grep` a number rather than retype it.
- Write floats as Python floats (no NumPy scalars — they crash `json.dump`).

---

## 7. Figure manifest (paper consumes these)

Result figures — produced by `analysis/*.py`, written to `figures/` as **both `.png` (200 dpi) and
`.pdf`**:

| file | content |
|---|---|
| `mapA_scatter` | 2-D PCA scatter, 400 Olivetti points, coloured by identity |
| `mapB_scatter` | same plot, same settings, Map B embeddings |
| `eigenfaces` | 16 eigenfaces, 4×4 grid (64×64 each) |
| `scree` | explained-variance ratio + cumulative for both maps (single panel, two colours) |
| `distance_histograms` | genuine vs impostor distance distributions, Map A and Map B |
| `roc_curves` | ROC for Map A and Map B with EER marked |
| `metric_bars` | silhouette / 1-NN / AUC / EER side-by-side, Map A vs Map B |
| `threshold_curves` | FAR & FRR vs threshold, Map A vs Map B |
| `live_heatmap` | 12×12 scan distance matrix with condition labels |
| `live_lighting` | noise floor vs intensity vs direction (bar with min–max whiskers) |
| `live_pooling` | frame-level vs pooled-scan distance distributions |
| `confusion_panel` | per-identity LOO accuracy Map A vs Map B scatter/bar |

Every figure: white background, no chartjunk, font sizes readable at column width (~3.3 in),
colourblind-safe palette, axes labelled with units, title states what is measured, PNG 200 dpi and a
vector PDF. Map A and Map B must share colour and axis conventions when compared.

Architecture diagrams — produced by the **paper** owner in `paper/figures/`, vector PDF (TikZ or
Graphviz `dot -Tpdf`):

| file | content |
|---|---|
| `fig_pipeline.pdf` | end-to-end system: data sources → Map A branch → Map B branch → evaluation; shows shared plotting/eval code and the frozen model |
| `fig_preprocess.pdf` | MTCNN cascade (P/R/O-Net) → bounding box + eye alignment → crop 160×160 → (x−127.5)/128; also the Olivetti shortcut path (no detector: pre-aligned, resize+replicate) |
| `fig_model.pdf` | InceptionResnetV1 block diagram with **verified** tensor shapes from `results/model_arch.json`; stem, 5×Inception-ResNet-A, 10×B, 5×C, reduction blocks, GAP, 512-d linear, L2-norm |
| `fig_protocol.pdf` | evaluation protocol: enrolment/probe split, genuine vs impostor trials, score distributions, ROC + threshold/FAR/FRR relation |
| `fig_lighting_setup.pdf` | the six capture conditions (front/left/right/bottom + room/flash) schematic |

Diagrams must be legible in grayscale, use consistent shapes/colours with the paper's palette, have
captions in the LaTeX source (not baked into the image), and be referenced from the text.

---

## 8. Paper specification

- LaTeX article, 11pt, `letterpaper`, `\usepackage[T1]{fontenc}`, `lmodern`, `microtype`,
  `booktabs`, `graphicx`, `hyperref`, `amsmath`, `xcolor`, `caption`, `subcaption`, `float`,
  `geometry` (margins 1 in). Compile with `pdflatex` (twice, for refs) from `paper/`.
- Working title: *Reading Identity from Pixels: A Controlled Comparison of Eigenfaces and Deep Face
  Embeddings, with a Webcam Case Study on Illumination*.
- Author: Nic Groenewald. No institutional affiliation; "Independent project". Date: 2026.
- Length target: 16 pages including figures and references. (The initial plan said 10–14; the
  figure manifest grew to 11 figure environments plus 3 tables, and trimming the diagnostics would
  remove evidence the paper argues from. 16 pages, no padding.)
- Sections: Abstract; 1 Introduction; 2 Background (Eigenfaces/PCA, deep FR, MTCNN, vggface2,
  metric learning + triplet loss); 3 Related work; 4 Data; 5 Method (pipeline, metrics, protocols);
  6 Results (Olivetti maps, quantitative comparison, live lighting study, threshold analysis);
  7 Discussion; 8 Limitations; 9 Reproducibility; 10 Future work (Yale B, LFW, triplet training);
  References; optional Appendix (full result tables).
- Every claim in the abstract/intro/results must match a number in `results/*.json`. No invented
  numbers, no "approximately" fudging. If a result is weak, say so.
- Report the limitations honestly: one subject, no stranger capture, Olivetti pre-aligned, tiny
  data, frozen model, webcam frames not independent, cross-domain thresholds not calibrated.
- The paper is about *what was actually done*: the Olivetti Eigenfaces study, the deep-embedding
  replication, the webcam lighting field study, and the threshold analysis. It is not a novel
  method paper — frame it as a controlled replication + measurement study with a practitioner's
  findings, and say so.
- References: **verify every one** with a web search (title/authors/venue/year/arXiv id). Use
  inline `thebibliography` or a `refs.bib` + natbib; either way the PDF must have no `[?]`.
- Use `\citep`/`\citet` consistently; list all cited works.
- Figures: include all result figures that support a claim; reference each in the text; captions
  self-contained (figure should be understandable without the body). Tables for the main
  quantitative comparison and threshold operating points.
- Write in third person, past tense for what was done, present tense for what is true. No hype, no
  "unprecedented", no claims the data can't support.
- Build must be warning-clean-ish (overfull boxes are tolerable; undefined refs are not).

---

## 9. Ownership / write scopes (hard rules)

| owner | may write | must not touch |
|---|---|---|
| **core-dev** | `fem/**`, `tests/**`, `scripts/live_capture.py`, `scripts/compare_scans.py`, `data/scans.json`, `pyproject.toml`, `Makefile`, `requirements.txt`, `.gitignore`, `docs/CODE.md`, `.cache/**` | `analysis/`, `results/`, `figures/`, `paper/`, `README.md`, findings docs, `docs/DESIGN.md` |
| **lab** | `analysis/**`, `results/**`, `figures/**`, `docs/findings/live-illumination.md`, `docs/findings/map-a-raw-pixels.md`, notebook edits under `notebooks/` | `fem/`, `tests/`, `scripts/live_capture.py`, `scripts/compare_scans.py`, `paper/`, `README.md` |
| **paper** | `paper/**` | everything else |
| **lead** | `README.md`, `docs/DESIGN.md`, `docs/REVIEW.md`, git commits, final integration | team files, except to fix a blocker with the owner's knowledge |

- **Only the Lead runs `git` write commands** (add/commit/tag/push). Teammates never commit.
- Raw data files (`data/scans/*.npy`, `data/`) are read-only for everyone except the
  Lead.
- If you need a file outside your scope, message the owner or the Lead; do not edit it.
- Report completion with the absolute paths you wrote and the commands you ran.
- Keep messages to the Lead short and factual: what changed, what's blocked, what you need.

## 10. Toolchain / commands

```sh
export MPLCONFIGDIR="$PWD/.cache/mpl"
PY=/Users/nic/.venvs/nn-crew/bin/python

$PY analysis/run_olivetti.py      # E1 + E2  -> results/olivetti.json, figures/*
$PY analysis/run_live.py          # E3       -> results/live.json, figures/*
$PY analysis/run_threshold.py     # E4       -> results/threshold.json, figures/threshold_curves.*
$PY analysis/model_arch.py        # E5       -> results/model_arch.json
$PY -m pytest -q                  # tests, no network, no webcam, < 60 s

cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

Runtime expectations: Olivetti embedding of 400 images is seconds on MPS; the whole pipeline under a
few minutes. The model weights are already cached; do not download anything else.
