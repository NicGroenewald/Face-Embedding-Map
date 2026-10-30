"""E1 + E2 — Map A (raw pixels) vs Map B (frozen InceptionResnetV1) on Olivetti.

Why this script exists
----------------------
Stages 2-4 of the plan ask one question with two pictures and one number: does a
learned embedding separate *identity* better than raw pixels do?  The pictures are
the 2-D PCA scatters; the numbers are computed in **full dimensionality** only
(silhouette, 1-NN leave-one-out, genuine/impostor ROC-AUC/EER, TAR@FAR).  The 2-D
plot is for eyes — measuring on it would be measuring plotting settings (plan trap
#3).

It also runs a full-rank PCA *isometry sanity check* (historical JSON key
``controls.map_a_pca512``): 400 centred examples span at most 399 dimensions,
so the requested 512-component pixel PCA cannot exist.  Projection onto all
399 observable components preserves pairwise distances; agreeing metrics
validate the arithmetic, but do not isolate feature count, intrinsic geometry,
architecture or pretraining as causes of the gap.

Everything numeric is delegated to ``fem`` (PCA, metrics) — this file only wires
the protocol and writes results/olivetti.json in the DESIGN.md §6 schema.

Run: ``$PY analysis/run_olivetti.py``
Outputs: results/olivetti.json, results/olivetti_maps.npz,
         results/embeddings_olivetti.npz (via fem.embed.embed_olivetti),
         figures/{mapA_scatter,mapB_scatter,eigenfaces,scree,metric_bars,
                  roc_curves,distance_histograms,confusion_panel}.{png,pdf}
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from analysis._common import (  # noqa: E402
    FIGURES_DIR,
    MAP_COLORS,
    RESULTS_DIR,
    SEED,
    common_header,
    describe_device,
    save_json,
)

# Protocol constants (DESIGN.md §5 E1/E2).
N_IDENTITIES = 40
N_IMAGES = 400
IMAGES_PER_IDENTITY = 10
PCA_2D = 2
SCREE_COMPONENTS = 100
CONTROL_COMPONENTS_REQUESTED = 512
GENUINE_EXPECTED = N_IDENTITIES * (IMAGES_PER_IDENTITY * (IMAGES_PER_IDENTITY - 1) // 2)
IMPOSTOR_EXPECTED = (
    N_IDENTITIES
    * (N_IDENTITIES - 1)
    // 2
    * IMAGES_PER_IDENTITY
    * IMAGES_PER_IDENTITY
)
COLORS = MAP_COLORS


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _flatten(images: np.ndarray) -> np.ndarray:
    """(N, 64, 64) -> (N, 4096) float64 (PCA workspace)."""
    return np.asarray(images, dtype=np.float64).reshape(len(images), -1)


def _roc_block(genuine: np.ndarray, impostor: np.ndarray, metrics_mod) -> dict:
    """AUC / EER / TAR@FAR / d' straight from fem.metrics.summary (one implementation)."""
    summary = metrics_mod.summary(genuine, impostor)
    return {
        "auc": summary["auc"],
        "eer": summary["eer"],
        "eer_threshold": summary["eer_threshold"],
        "tar_at_far_1e-3": summary["tar_at_far_1e-3"],
        "tar_at_far_1e-2": summary["tar_at_far_1e-2"],
        "d_prime": summary["d_prime"],
    }


def _full_metrics(X: np.ndarray, labels: np.ndarray, coords: np.ndarray, metrics_mod, maps_mod) -> dict:
    """All Stage-4 numbers for one representation, measured on FULL-dimensional X.

    ``coords`` is the 2-D PCA projection of X, used only for the geometry-fidelity
    check.  Never for silhouette/kNN/ROC.
    """
    silhouette = float(metrics_mod.silhouette(X, labels))
    knn = metrics_mod.knn_loo(X, labels, k=1)
    predictions = np.asarray(knn["predictions"], dtype=int)

    genuine, impostor = metrics_mod.pair_distances(X, labels)
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    assert genuine.size == GENUINE_EXPECTED, f"genuine pairs {genuine.size} != {GENUINE_EXPECTED}"
    assert impostor.size == IMPOSTOR_EXPECTED, f"impostor pairs {impostor.size} != {IMPOSTOR_EXPECTED}"
    # Sanity invariants that could silently invert the whole result if wrong.
    assert genuine.mean() < impostor.mean(), "genuine distances are not smaller than impostor distances"

    summary = metrics_mod.summary(genuine, impostor)
    return {
        "silhouette": silhouette,
        "knn_loo": {"accuracy": float(knn["accuracy"]), "top5": float(knn["top5"])},
        "pairs": {"genuine": summary["genuine"], "impostor": summary["impostor"]},
        "roc": _roc_block(genuine, impostor, metrics_mod),
        "rank_correlation_2d": float(maps_mod.rank_correlation_2d(coords, X)),
        "predictions": predictions,  # internal, stripped before JSON
        "per_identity_knn_accuracy": [
            float(v) for v in metrics_mod.per_identity_accuracy(labels, predictions)
        ],
        "_genuine": genuine,
        "_impostor": impostor,
    }


def _seed_robustness(X: np.ndarray, maps_mod, metrics_mod) -> dict:
    """Does the 2-D picture depend on the PCA random seed? (Plan Stage 4 'also'.)

    Full-dimensional metrics are deterministic (silhouette/kNN are distance-only);
    what can move is the randomized-SVD 2-D projection.  We report (a) each seed's
    2-D-vs-full Spearman geometry fidelity and (b) rank correlation between the
    seeds' 2-D geometries, computed on pairwise-distance vectors so an axis sign
    flip (which is meaningless) cannot fake a disagreement.
    """
    seeds = [0, 1, 2]
    coord_list = []
    fidelity = []
    for seed in seeds:
        pca = maps_mod.pca_map(X, n_components=PCA_2D, seed=seed)
        coord_list.append(np.asarray(pca.coords, dtype=np.float64))
        fidelity.append(float(maps_mod.rank_correlation_2d(pca.coords, X)))

    dist_vectors = []
    for coords in coord_list:
        full = np.asarray(metrics_mod.pairwise_distances(coords), dtype=np.float64)
        rows, cols = np.triu_indices_from(full, k=1)
        dist_vectors.append(full[rows, cols])

    matrix = np.eye(len(seeds))
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            rho = float(spearmanr(dist_vectors[i], dist_vectors[j]).statistic)
            matrix[i, j] = matrix[j, i] = rho

    return {
        "seeds": seeds,
        "spearman_2d_vs_full": fidelity,
        "pairwise_2d_rank_corr": matrix.tolist(),
        "note": "pairwise_2d_rank_corr is Spearman over the 2-D pairwise-distance "
        "vectors between seeds, invariant to meaningless axis sign flips/rotations. "
        "Full-dimensional metrics are deterministic and seed-independent.",
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> dict:
    started = time.time()
    device = describe_device()
    print(f"[olivetti] device={device}")

    # Imported here so the module-level env vars (MPLCONFIGDIR) are set first.
    from fem import metrics as fem_metrics
    from fem import maps as fem_maps
    from fem.data import load_olivetti
    from fem.embed import FaceEmbedder, embed_olivetti
    from fem.plotting import (
        save_confusion_panel,
        save_distance_histograms,
        save_eigenfaces,
        save_metric_bars,
        save_roc,
        save_scatter,
        save_scree,
    )
    from fem.preprocess import to_model_input

    results: dict = {}
    notes = [
        "The Map A and Map B 2-D PCA plots are fitted on all 400 images, including "
        "future verification probes. This transductive fit affects the visualisations, "
        "not the headline metrics, which use unprojected full-dimensional vectors; "
        "it cannot establish that the Map B advantage is conservative.",
        "All clustering/verification metrics are computed in full dimensionality "
        "(4096-d pixels / 512-d embeddings); the 2-D PCA scatters are for "
        "visualisation only and are never measured on.",
    ]

    # ---------------- data ----------------
    images, labels = load_olivetti()
    labels = np.asarray(labels, dtype=int)
    assert images.shape == (N_IMAGES, 64, 64), f"unexpected Olivetti shape {images.shape}"
    assert labels.shape == (N_IMAGES,) and set(np.unique(labels)) == set(range(N_IDENTITIES))
    assert np.bincount(labels).tolist() == [IMAGES_PER_IDENTITY] * N_IDENTITIES

    X_a = _flatten(images)
    pixel_range = [float(images.min()), float(images.max())]
    print(f"[olivetti] images {images.shape}, pixel range {pixel_range}")

    # ---------------- E1: Map A ----------------
    print("[olivetti] E1: Map A (raw pixels)")
    pca2_a = fem_maps.pca_map(X_a, n_components=PCA_2D, seed=SEED)
    pca_hi_a = fem_maps.pca_map(X_a, n_components=SCREE_COMPONENTS, seed=SEED)
    evr_a = np.asarray(pca_hi_a.evr, dtype=np.float64)
    milestones_a = fem_maps.variance_milestones(evr_a)
    metrics_a = _full_metrics(X_a, labels, pca2_a.coords, fem_metrics, fem_maps)
    seed_a = _seed_robustness(X_a, fem_maps, fem_metrics)
    print(
        f"[olivetti] Map A: silhouette={metrics_a['silhouette']:.4f} "
        f"1NN={metrics_a['knn_loo']['accuracy']:.4f} "
        f"AUC={metrics_a['roc']['auc']:.6f} EER={metrics_a['roc']['eer']:.4f} "
        f"d'={metrics_a['roc']['d_prime']:.3f} "
        f"2D-rank-corr={metrics_a['rank_correlation_2d']:.4f}"
    )

    # ---------------- E2: Map B ----------------
    print("[olivetti] E2: Map B (InceptionResnetV1 vggface2)")
    embedder = FaceEmbedder(device=device)
    embeddings = embed_olivetti(embedder, cache=RESULTS_DIR / "embeddings_olivetti.npz")
    embeddings = np.asarray(embeddings, dtype=np.float64)
    assert embeddings.shape == (N_IMAGES, 512), f"unexpected embedding shape {embeddings.shape}"
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), (
        f"embeddings are not unit-norm: min={norms.min():.6f} max={norms.max():.6f}"
    )

    # Preprocessing is the silent-invalidator: assert the model input really is the
    # MTCNN-style standardised 160x160 RGB tensor and not raw [0,1] or un-replicated grey.
    model_input = to_model_input(images)
    input_stats = _verify_model_input(model_input, images)
    print(f"[olivetti] model input: {input_stats}")

    pca2_b = fem_maps.pca_map(embeddings, n_components=PCA_2D, seed=SEED)
    pca_hi_b = fem_maps.pca_map(embeddings, n_components=SCREE_COMPONENTS, seed=SEED)
    evr_b = np.asarray(pca_hi_b.evr, dtype=np.float64)
    milestones_b = fem_maps.variance_milestones(evr_b)
    metrics_b = _full_metrics(embeddings, labels, pca2_b.coords, fem_metrics, fem_maps)
    seed_b = _seed_robustness(embeddings, fem_maps, fem_metrics)
    print(
        f"[olivetti] Map B: silhouette={metrics_b['silhouette']:.4f} "
        f"1NN={metrics_b['knn_loo']['accuracy']:.4f} "
        f"AUC={metrics_b['roc']['auc']:.6f} EER={metrics_b['roc']['eer']:.4f} "
        f"d'={metrics_b['roc']['d_prime']:.3f} "
        f"2D-rank-corr={metrics_b['rank_correlation_2d']:.4f}"
    )

    # ---------------- matched-dimensionality control ----------------
    control = _full_rank_isometry_check(
        X_a, labels, metrics_a, fem_maps, fem_metrics
    )
    print(
        f"[olivetti] control: PCA {control['components']} comps "
        f"(requested {CONTROL_COMPONENTS_REQUESTED}), variance retained "
        f"{control['variance_retained']:.9f}, silhouette {control['silhouette']:.6f} "
        f"(Map A {metrics_a['silhouette']:.6f})"
    )

    # ---------------- confusion analysis ----------------
    pred_a = np.asarray(metrics_a["predictions"], dtype=int)
    pred_b = np.asarray(metrics_b["predictions"], dtype=int)
    confusion = _confusion(pred_a, pred_b, labels, metrics_a, metrics_b)
    print(
        f"[olivetti] confusion: same-prediction {confusion['same_prediction_fraction']:.4f}, "
        f"per-identity r={confusion['per_identity_accuracy_corr_pearson']}, "
        f"both-wrong-same-choice={confusion['both_wrong_same_choice_fraction']}"
    )
    if confusion["both_wrong_count"] == 0:
        notes.append(
            "confusion.both_wrong_same_choice_fraction is null because Map B's 1-NN "
            "accuracy is 100%: no image is misclassified by both maps, so the "
            "fraction has no denominator. Reported as null rather than 0."
        )

    # Extra methodological caveats for the paper.
    notes.append(control["note"])
    notes.append(
        "dataset.pixel_range is the dataset-wide range [0.0, 1.0] over all 400 images. "
        "The old Map A findings quote 0.083-0.876, which is the range of image 0 only; "
        "the dataset as a whole spans the full [0, 1] scale."
    )
    if confusion["per_identity_accuracy_corr_pearson"] is None:
        notes.append(
            "confusion.per_identity_accuracy_corr_pearson is null because one map's "
            "per-identity accuracy is constant (Map B is 100%), so Pearson correlation "
            "is undefined. Reported as null rather than 0."
        )

    # Which pixel statistic does each of the two plotted axes actually track?
    # This is the numeric backing for the "2-D plots mislead" claim: the lobes in
    # the Map A scatter are illumination, not identity.
    axes_interpretation = _pca_axes_interpretation(pca2_a.coords, images)
    print(
        "[olivetti] PCA axes vs pixel statistics: "
        f"PC1~brightness r={axes_interpretation['pc1_vs_mean_brightness_pearson']:.4f} "
        f"(rho={axes_interpretation['pc1_vs_mean_brightness_spearman']:.4f}), "
        f"PC2~lateral r={axes_interpretation['pc2_vs_lateral_brightness_pearson']:.4f}"
    )

    # ---------------- JSON payload (DESIGN.md §6 schema) ----------------
    payload = {
        "common": common_header(device, notes=notes),
        "dataset": {
            "name": "Olivetti",
            "n_images": int(N_IMAGES),
            "n_identities": int(N_IDENTITIES),
            "images_per_identity": int(IMAGES_PER_IDENTITY),
            "image_shape": [64, 64],
            "pixel_range": pixel_range,
            "aligned": "pre-aligned, tightly cropped (documented limitation)",
        },
        "map_a": _map_json("raw pixels (4096-d)", evr_a, milestones_a, metrics_a, seed_a)
        | {
            "pc1_interpretation": _pc1_interpretation(),
            "pca_axes_interpretation": axes_interpretation,
        },
        "map_b": _map_json("InceptionResnetV1 vggface2 (512-d unit)", evr_b, milestones_b, metrics_b, seed_b)
        | {
            "pc1_interpretation": "not a pixel image; components are directions in embedding space",
            "input_stats": input_stats,
        },
        "controls": {"map_a_pca512": control},
        "confusion": confusion,
    }
    payload["controls"]["map_a_pca512"].pop("_internal", None)

    # ---------------- figures ----------------
    print("[olivetti] figures")
    save_scatter(
        pca2_a.coords,
        labels,
        FIGURES_DIR / "mapA_scatter",
        title="Map A: raw pixels, PCA to 2-D",
    )
    save_scatter(
        pca2_b.coords,
        labels,
        FIGURES_DIR / "mapB_scatter",
        title="Map B: deep embeddings, PCA to 2-D",
    )
    save_eigenfaces(pca_hi_a.model.components_, FIGURES_DIR / "eigenfaces", n=16)
    save_scree(
        {"Map A": evr_a, "Map B": evr_b},
        FIGURES_DIR / "scree",
        title="Explained variance: Map A (4096-d pixels) vs Map B (512-d embeddings)",
    )
    save_distance_histograms(
        {"Map A": (metrics_a["_genuine"], metrics_a["_impostor"]),
         "Map B": (metrics_b["_genuine"], metrics_b["_impostor"])},
        FIGURES_DIR / "distance_histograms",
    )
    save_roc(
        {"Map A": (metrics_a["_genuine"], metrics_a["_impostor"]),
         "Map B": (metrics_b["_genuine"], metrics_b["_impostor"])},
        FIGURES_DIR / "roc_curves",
    )
    save_metric_bars(_metric_bars(metrics_a, metrics_b), FIGURES_DIR / "metric_bars")
    save_confusion_panel(
        metrics_a["per_identity_knn_accuracy"],
        metrics_b["per_identity_knn_accuracy"],
        FIGURES_DIR / "confusion_panel",
        labels=[str(i) for i in range(N_IDENTITIES)],
    )

    # ---------------- npz artefacts ----------------
    np.savez(
        RESULTS_DIR / "olivetti_maps.npz",
        coords_a=np.asarray(pca2_a.coords, dtype=np.float32),
        coords_b=np.asarray(pca2_b.coords, dtype=np.float32),
        labels=labels.astype(np.int16),
        evr_a=evr_a.astype(np.float32),
        evr_b=evr_b.astype(np.float32),
        components_a=np.asarray(pca_hi_a.model.components_, dtype=np.float32),
        components_b=np.asarray(pca_hi_b.model.components_, dtype=np.float32),
    )

    # Strip internal-only arrays before writing JSON.
    for block in ("map_a", "map_b"):
        payload[block].pop("predictions", None)

    out_path = RESULTS_DIR / "olivetti.json"
    save_json(payload, out_path)
    elapsed = time.time() - started
    print(f"[olivetti] wrote {out_path} in {elapsed:.1f}s")
    return payload


# --------------------------------------------------------------------------- #
# helpers that need the fem import to exist at call time
# --------------------------------------------------------------------------- #
def _verify_model_input(model_input, images: np.ndarray) -> dict:
    """Assert the tensor fed to Map B is the mandatory MTCNN-style preprocessing.

    A silent preprocessing mistake is the one bug that invalidates Map B while
    still producing plausible-looking numbers, so these are hard asserts:
      * shape (400, 3, 160, 160), float32;
      * greyscale was replicated to 3 identical channels;
      * values are standardised: (x*255 - 127.5)/128, i.e. range well outside [0, 1]
        and centred near 0 — a [0,1] tensor or a mean near 0.5 means the
        standardisation was skipped;
      * it matches an independent reference implementation of the same recipe.
    """
    import torch
    import torch.nn.functional as F

    x = model_input.detach().cpu()
    assert tuple(x.shape) == (N_IMAGES, 3, 160, 160), f"model input shape {tuple(x.shape)}"
    assert x.dtype == torch.float32, f"model input dtype {x.dtype}"
    assert torch.equal(x[:, 0], x[:, 1]) and torch.equal(x[:, 1], x[:, 2]), (
        "greyscale image was not replicated identically into 3 channels"
    )

    stats = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "channels_identical": True,
    }
    assert stats["min"] < -0.5 and stats["max"] > 0.5, (
        "model input looks un-standardised (range inside [0,1]): "
        f"min={stats['min']:.3f} max={stats['max']:.3f}"
    )
    assert -1.0 <= stats["min"] <= stats["max"] <= 1.0, "standardised input outside [-1, 1]"
    assert abs(stats["mean"]) < 0.5, f"model input mean {stats['mean']:.3f} is not near 0"

    # Independent reference: replicate -> bilinear antialiased resize -> *255 -> standardise.
    ref = torch.as_tensor(images, dtype=torch.float32).unsqueeze(1).repeat(1, 3, 1, 1)
    ref = F.interpolate(ref, size=(160, 160), mode="bilinear", align_corners=False, antialias=True)
    ref = (ref * 255.0 - 127.5) / 128.0
    max_abs_diff = float((x - ref).abs().max())
    stats["reference_recipe_max_abs_diff"] = max_abs_diff
    assert max_abs_diff < 1e-4, (
        f"fem.preprocess.to_model_input differs from the reference recipe by {max_abs_diff:.2e}; "
        "check resize mode/antialias and standardisation"
    )
    return stats


def _full_rank_isometry_check(
    X_a: np.ndarray, labels: np.ndarray, metrics_a: dict, maps_mod, metrics_mod
) -> dict:
    """Verify that full-rank pixel PCA preserves Euclidean distances.

    Requested 512 components are impossible for 400 centred samples (rank <=399).
    Keeping all 399 observable components is an orthogonal isometry, so the
    output should reproduce raw-pixel metrics numerically. This sanity check
    does not control for dimensionality or identify why Map B wins.
    """
    rank_limit = N_IMAGES - 1  # 400 centred samples live in a 399-dim subspace
    pca_ctrl = maps_mod.pca_map(X_a, n_components=rank_limit, seed=SEED)
    ctrl = np.asarray(pca_ctrl.coords, dtype=np.float64)
    variance_retained = float(np.asarray(pca_ctrl.evr, dtype=np.float64).sum())

    ctrl_2d = maps_mod.pca_map(ctrl, n_components=PCA_2D, seed=SEED)
    metrics = _full_metrics(ctrl, labels, ctrl_2d.coords, metrics_mod, maps_mod)

    # Numerically verify the orthogonal-invariance claim instead of asserting it.
    raw_d = np.asarray(metrics_mod.pairwise_distances(X_a), dtype=np.float64)
    ctrl_d = np.asarray(metrics_mod.pairwise_distances(ctrl), dtype=np.float64)
    rows, cols = np.triu_indices_from(raw_d, k=1)
    max_abs_distance_diff = float(np.abs(raw_d[rows, cols] - ctrl_d[rows, cols]).max())
    assert max_abs_distance_diff < 1e-3, (
        f"PCA projection changed pairwise distances by {max_abs_distance_diff:.3e}; "
        "the full-rank PCA isometry check failed"
    )

    return {
        "space": f"raw pixels projected to {rank_limit} PCA components",
        "components": int(rank_limit),
        "requested_components": CONTROL_COMPONENTS_REQUESTED,
        "variance_retained": variance_retained,
        "silhouette": metrics["silhouette"],
        "knn_loo": metrics["knn_loo"],
        "pairs": metrics["pairs"],
        "roc": metrics["roc"],
        "rank_correlation_2d": metrics["rank_correlation_2d"],
        "max_abs_distance_diff_vs_raw": max_abs_distance_diff,
        "note": (
            "512 components were requested, but 400 centred samples have rank <=399. "
            "A full-rank 399-component PCA preserves raw-pixel pairwise distances "
            "and therefore reproduces its metrics (up to numerical precision). "
            "This is an isometry sanity check, not a causal dimensionality ablation: "
            "both observed point clouds have sample rank <=399 and the model's "
            "pretraining, geometry and nominal widths remain confounded."
        ),
    }


def _confusion(pred_a: np.ndarray, pred_b: np.ndarray, labels: np.ndarray,
               metrics_a: dict, metrics_b: dict) -> dict:
    """Are the same people confused by both maps? (E2 step 8.)"""
    acc_a = np.asarray(metrics_a["per_identity_knn_accuracy"], dtype=np.float64)
    acc_b = np.asarray(metrics_b["per_identity_knn_accuracy"], dtype=np.float64)
    if np.ptp(acc_a) == 0 or np.ptp(acc_b) == 0:
        pearson = None
    else:
        pearson = float(np.corrcoef(acc_a, acc_b)[0, 1])

    both_wrong = (pred_a != labels) & (pred_b != labels)
    both_wrong_count = int(both_wrong.sum())
    if both_wrong_count:
        same_wrong = float((pred_a[both_wrong] == pred_b[both_wrong]).mean())
    else:
        same_wrong = None

    return {
        "per_identity_accuracy_corr_pearson": pearson,
        "same_prediction_fraction": float((pred_a == pred_b).mean()),
        "both_wrong_same_choice_fraction": same_wrong,
        "same_prediction_count": int((pred_a == pred_b).sum()),
        "both_wrong_count": both_wrong_count,
        "n_images": int(len(labels)),
    }


def _map_json(space, evr, milestones, metrics, seed_robustness) -> dict:
    """One map's block, matching DESIGN.md §6 exactly (plus defined internals)."""
    return {
        "space": space,
        "variance_milestones": {str(k): v for k, v in milestones.items()},
        "component0_evr": float(evr[0]),
        "silhouette": metrics["silhouette"],
        "knn_loo": metrics["knn_loo"],
        "pairs": metrics["pairs"],
        "roc": metrics["roc"],
        "rank_correlation_2d": metrics["rank_correlation_2d"],
        "seed_robustness": seed_robustness,
        "per_identity_knn_accuracy": metrics["per_identity_knn_accuracy"],
        "predictions": metrics["predictions"],
    }


def _metric_bars(metrics_a: dict, metrics_b: dict) -> dict:
    """Grouped-bar input: metric -> (Map A value, Map B value)."""
    return {
        "silhouette": (metrics_a["silhouette"], metrics_b["silhouette"]),
        "1-NN LOO accuracy": (
            metrics_a["knn_loo"]["accuracy"],
            metrics_b["knn_loo"]["accuracy"],
        ),
        "ROC-AUC": (metrics_a["roc"]["auc"], metrics_b["roc"]["auc"]),
        "EER": (metrics_a["roc"]["eer"], metrics_b["roc"]["eer"]),
    }


def _pca_axes_interpretation(coords: np.ndarray, images: np.ndarray) -> dict:
    """Correlate the seed-0 2-D PCA coordinates with simple per-image pixel statistics.

    Why: the paper's "2-D plots mislead" claim needs a number, not a vibe.  If PC1
    is essentially overall brightness and PC2 the left-minus-right brightness
    gradient, then the two visible lobes in the Map A scatter are dark vs bright
    photos, not identities.  Pearson is the direct linear correlation; Spearman is
    reported too because it makes no linearity assumption (the relationship between
    a projection coordinate and a brightness statistic need not be exactly linear).

    Only Map A gets this: the Map B coordinates are directions in embedding space
    with no simple pixel-statistic meaning.
    """
    coords = np.asarray(coords, dtype=np.float64)
    images = np.asarray(images, dtype=np.float64)
    if coords.shape != (len(images), 2):
        raise ValueError(f"expected ({len(images)}, 2) coords, got {coords.shape}")

    flattened = images.reshape(len(images), -1)
    brightness = flattened.mean(axis=1)
    half = images.shape[2] // 2
    lateral = images[:, :, :half].mean(axis=(1, 2)) - images[:, :, half:].mean(axis=(1, 2))

    pc1, pc2 = coords[:, 0], coords[:, 1]
    return {
        "pc1_vs_mean_brightness_pearson": float(pearsonr(pc1, brightness).statistic),
        "pc1_vs_mean_brightness_spearman": float(spearmanr(pc1, brightness).statistic),
        "pc2_vs_lateral_brightness_pearson": float(pearsonr(pc2, lateral).statistic),
        "note": "2-D PCA coordinates (seed 0) correlated with per-image pixel statistics "
        "on the 400 Olivetti images; PC1 tracks overall brightness, PC2 tracks the "
        "left-right illumination gradient.",
    }


def _pc1_interpretation() -> str:
    """Qualitative reading of the leading eigenfaces (see figures/eigenfaces.*)."""
    return (
        "leading components are illumination gradients — component 0 a broad "
        "brightness gradient, component 1 left-right brightness (qualitative reading "
        "of the component images in figures/eigenfaces)"
    )


if __name__ == "__main__":
    main()
