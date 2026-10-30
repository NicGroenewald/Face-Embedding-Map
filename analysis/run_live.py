"""E3 — the live lighting study: 12 webcam scans x 10 frames, one subject.

Why this script exists
----------------------
Stage 5 of the plan, done on a real face: the same person, one camera, six
lighting conditions, two repeats each.  This is an *observational, single-subject*
study of a **fixed historical 12-scan protocol**; it measures how far an embedding
moves when nothing changed (frame jitter), when the light was brighter (intensity),
and when the light came from a different direction.  The ratio direction/noise is
a descriptive "lighting direction costs N x frame jitter".

It cannot attribute differences to a single cause: brightness, direction and any
incidental change in pose or position were not independently manipulated, and the
scans were captured once.  Language here is deliberately non-causal.  The 12 scans
are a fixed design; new captures stay private and are not picked up automatically
without a redesign (and a schema change for a larger matrix).

The data already exists as data/scans/*.npy (unit vectors, MTCNN-cropped);
no model re-run is needed.  ``fem.data.load_scans()`` reads the manifest that records
which file is which condition.

Two extra measurements describe how much the 10 frames can be trusted: temporal
autocorrelation (frame_lag) and a first-frame outlier check (frame_outlier) — both
feed the limitation that frames are not independent samples.

Run: ``$PY analysis/run_live.py``
Outputs: results/live.json;
         figures/{live_heatmap,live_lighting,live_pooling}.{png,pdf}
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from analysis._common import (  # noqa: E402
    FIGURES_DIR,
    RESULTS_DIR,
    common_header,
    describe_device,
    save_json,
)

N_SCANS = 12
N_FRAMES = 10
# Fixed display/analysis order: room first, then flash, brightest to darkest side.
CONDITION_ORDER = ["room/front", "room/left", "flash/front", "flash/left", "flash/right", "flash/bottom"]
FLASH_DIRS = ["front", "left", "right", "bottom"]
DIRECTION_PAIRS = [
    ("front", "left"),
    ("front", "right"),
    ("front", "bottom"),
    ("left", "right"),
    ("left", "bottom"),
    ("right", "bottom"),
]
CROSS_DOMAIN_CAVEAT = (
    "different capture/crop/resolution; not a calibrated impostor distribution, and "
    "neither an upper nor a lower bound on webcam-domain stranger distance"
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _stats(values: np.ndarray) -> dict:
    """min/mean/max in plain Python floats (JSON-safe)."""
    values = np.asarray(values, dtype=np.float64).ravel()
    return {"min": float(values.min()), "mean": float(values.mean()), "max": float(values.max())}


def _cos_from_euclidean(d: float) -> float:
    """For unit vectors cos(theta) = 1 - d^2/2 (DESIGN.md §5)."""
    return float(1.0 - (d * d) / 2.0)


def _cos_block(d: float) -> dict:
    """Both interpretations of one headline distance."""
    return {
        "euclidean": float(d),
        "cosine_similarity": _cos_from_euclidean(d),
        "cosine_distance": float((d * d) / 2.0),
    }


def _condition(scan) -> str:
    return f"{scan.intensity}/{scan.direction}"


def _ordered_scans(scan_set) -> tuple[list, list[int]]:
    """Index the fixed historical 12 scans into condition-then-repeat order.

    Fixed design: exactly 12 scans and exactly 2 repeats per condition.  A
    manifest with extra scans is rejected loudly rather than silently mixed into
    the historical analysis — new captures require a deliberate redesign (and a
    schema change for a larger matrix), which this script does not do.

    ``Scan`` carries no repeat field (DESIGN §4), so repeats are ordered by name,
    which the manifest fixes as '<condition>_<repeat>'.
    """
    scans = list(scan_set.scans)
    if len(scans) != N_SCANS:
        raise ValueError(
            f"fixed historical protocol expects exactly {N_SCANS} scans, got {len(scans)}; "
            "new captures are not picked up automatically and require a redesign"
        )
    ordered = []
    for condition in CONDITION_ORDER:
        group = [s for s in scans if _condition(s) == condition]
        if len(group) != 2:
            raise ValueError(
                f"fixed historical protocol expects 2 scans for {condition}, got {len(group)}"
            )
        ordered.extend(sorted(group, key=lambda s: s.name))
    return ordered, [scans.index(s) for s in ordered]


def _frame_pair_values(a: np.ndarray, b: np.ndarray, fem_metrics, within: bool = False) -> np.ndarray:
    """All 10x10 Euclidean distances between two scans' frames, or the 45 within one scan."""
    if within:
        d = fem_metrics.pairwise_distances(a)
        rows, cols = np.triu_indices(a.shape[0], k=1)
        return np.asarray(d[rows, cols], dtype=np.float64)
    return np.asarray(fem_metrics.pairwise_distances(a, b), dtype=np.float64).ravel()


def _load_olivetti_embeddings() -> np.ndarray:
    """Read only the validated Olivetti cache from the same model/data recipe.

    A numeric `(400, 512)` array alone is not sufficient: the shared loader
    checks checkpoint, dataset, preprocessing and row integrity, and gives an
    actionable `make maps` message if the cache is absent or stale.
    """
    from fem.embed import load_cached_olivetti_embeddings

    return np.asarray(load_cached_olivetti_embeddings(), dtype=np.float64)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> dict:
    started = time.time()
    device = describe_device()

    from fem.data import load_scans
    from fem import metrics as fem_metrics
    from fem.plotting import save_heatmap, save_lighting_panel, save_pooling

    scan_set = load_scans()
    ordered, order_idx = _ordered_scans(scan_set)
    names = [s.name for s in ordered]
    conditions = [_condition(s) for s in ordered]
    stack = np.asarray(scan_set.stack(), dtype=np.float64)[order_idx]  # (12,10,512)
    assert stack.shape == (N_SCANS, N_FRAMES, 512), f"unexpected scan stack {stack.shape}"
    norms = np.linalg.norm(stack, axis=2)
    assert np.allclose(norms, 1.0, atol=1e-4), (
        f"saved live embeddings are not unit-norm: {norms.min():.6f}..{norms.max():.6f}"
    )
    print(f"[live] {len(ordered)} scans x {N_FRAMES} frames, unit norms ok (device {device})")

    # ---------------- 1. within-scan jitter: the noise floor ----------------
    within = []
    for k in range(N_SCANS):
        values = _frame_pair_values(stack[k], stack[k], fem_metrics, within=True)
        assert values.size == N_FRAMES * (N_FRAMES - 1) // 2, f"expected 45 pairs, got {values.size}"
        within.append({"scan": names[k], **_stats(values)})
    noise_floor_mean = float(np.mean([w["mean"] for w in within]))
    print(f"[live] within-scan jitter mean {noise_floor_mean:.4f} "
          f"(range of per-scan means {min(w['mean'] for w in within):.4f}..{max(w['mean'] for w in within):.4f})")

    # ---------------- full frame-pair distance matrix ----------------
    matrix = np.zeros((N_SCANS, N_SCANS), dtype=np.float64)
    cross_values: dict[tuple[int, int], np.ndarray] = {}
    for i in range(N_SCANS):
        matrix[i, i] = within[i]["mean"]
        for j in range(i + 1, N_SCANS):
            values = _frame_pair_values(stack[i], stack[j], fem_metrics)
            assert values.size == 100, f"expected 100 frame pairs for scans {i},{j}"
            cross_values[(i, j)] = values
            matrix[i, j] = matrix[j, i] = float(values.mean())

    # Re-normalised scan means: the pooling representation ("enrol 10 frames,
    # average them").  Renormalised so distances stay between unit vectors.
    scan_means = np.asarray(scan_set.mean_embeddings(), dtype=np.float64)[order_idx]
    assert scan_means.shape == (N_SCANS, 512)
    assert np.allclose(np.linalg.norm(scan_means, axis=1), 1.0, atol=1e-5)
    raw_mean_norms = np.linalg.norm(stack.mean(axis=1), axis=1)

    # ---------------- 2. same-condition between-scan ----------------
    same_condition = []
    for condition in CONDITION_ORDER:
        idx = [k for k, c in enumerate(conditions) if c == condition]
        assert len(idx) == 2
        values = cross_values[tuple(sorted(idx))]
        same_condition.append({"condition": condition, **_stats(values)})
    print("[live] same-condition between-scan means: " +
          ", ".join(f"{s['condition']}={s['mean']:.3f}" for s in same_condition))

    # ---------------- 3. observational brightness comparison (front light only) ----------------
    room_front = [k for k, c in enumerate(conditions) if c == "room/front"]
    flash_front = [k for k, c in enumerate(conditions) if c == "flash/front"]
    intensity_values = np.concatenate(
        [cross_values[tuple(sorted((i, j)))] for i in room_front for j in flash_front]
    )
    intensity_effect = {
        "condition_pair": ["room/front", "flash/front"],
        **_stats(intensity_values),
        "note": f"observational comparison: mean {intensity_values.mean():.3f} sits within "
                f"the same-condition between-scan means "
                f"{min(s['mean'] for s in same_condition):.3f}-"
                f"{max(s['mean'] for s in same_condition):.3f} (within-scan jitter mean "
                f"{noise_floor_mean:.3f}). Room-front and flash-front were captured in "
                f"separate scans, so this design cannot attribute any difference to "
                f"brightness alone and makes no causal claim.",
    }
    print(f"[live] observational brightness comparison (room front vs flash front): "
          f"mean {intensity_values.mean():.4f}")

    # ---------------- 4. direction effect (flash only, six pairs) ----------------
    direction_pairs = []
    for d1, d2 in DIRECTION_PAIRS:
        i1 = [k for k, c in enumerate(conditions) if c == f"flash/{d1}"]
        i2 = [k for k, c in enumerate(conditions) if c == f"flash/{d2}"]
        values = np.concatenate(
            [cross_values[tuple(sorted((i, j)))] for i in i1 for j in i2]
        )
        direction_pairs.append({"pair": [f"flash/{d1}", f"flash/{d2}"], **_stats(values)})
    direction_effect_mean = float(np.mean([p["mean"] for p in direction_pairs]))
    direction_effect = {"pairs": direction_pairs, "mean": direction_effect_mean}
    direction_to_noise_ratio = float(direction_effect_mean / noise_floor_mean)
    print(f"[live] direction effect mean {direction_effect_mean:.4f} -> "
          f"{direction_to_noise_ratio:.2f}x the noise floor")

    # ---------------- 5. pooling effect ----------------
    frame_level = np.concatenate(list(cross_values.values()))  # 66 scan pairs x 100 frames
    pooled = np.asarray(
        fem_metrics.pairwise_distances(scan_means), dtype=np.float64
    )
    rows, cols = np.triu_indices(N_SCANS, k=1)
    pooled_values = pooled[rows, cols]
    assert pooled_values.size == 66
    reduction_pct = float((frame_level.mean() - pooled_values.mean()) / frame_level.mean() * 100.0)
    pooling = {
        "frame_level_mean": float(frame_level.mean()),
        "scan_mean_level_mean": float(pooled_values.mean()),
        "reduction_pct": reduction_pct,
        "frame_level_n": int(frame_level.size),
        "scan_mean_level_n": int(pooled_values.size),
        "raw_scan_mean_norm_before_renormalisation": {
            "min": float(raw_mean_norms.min()),
            "mean": float(raw_mean_norms.mean()),
            "max": float(raw_mean_norms.max()),
        },
    }
    print(f"[live] pooling: {frame_level.mean():.4f} -> {pooled_values.mean():.4f} "
          f"({reduction_pct:.1f}% reduction)")

    # ---------------- 6. worst genuine distance ----------------
    worst_value, worst_pair = -1.0, None
    for (i, j), values in cross_values.items():
        if values.max() > worst_value:
            worst_value, worst_pair = float(values.max()), (i, j)
    wi, wj = worst_pair
    worst_scan_mean = float(np.linalg.norm(scan_means[wi] - scan_means[wj]))
    worst_scan_mean_overall = float(pooled_values.max())
    sm_rows, sm_cols = np.triu_indices(N_SCANS, k=1)
    overall_pos = int(np.argmax(pooled_values))
    sm_pair = (int(sm_rows[overall_pos]), int(sm_cols[overall_pos]))
    worst = {
        "frame_level": worst_value,
        "scan_mean_level": worst_scan_mean,
        "pair": [names[wi], names[wj]],
        "cosine_equivalent": _cos_from_euclidean(worst_value),
        "cosine_distance": float(worst_value ** 2 / 2.0),
        "worst_scan_mean_pair": {
            "pair": [names[sm_pair[0]], names[sm_pair[1]]],
            "distance": worst_scan_mean_overall,
            "cosine_similarity": _cos_from_euclidean(worst_scan_mean_overall),
        },
        "frame_level_n": 100,
    }
    print(f"[live] worst genuine frame pair {worst_value:.4f} between {worst['pair']} "
          f"(cos {worst['cosine_equivalent']:.3f}); same pair pooled {worst_scan_mean:.4f}; "
          f"worst pooled pair {worst_scan_mean_overall:.4f} {worst['worst_scan_mean_pair']['pair']}")

    # ---------------- 7. cross-domain sanity check ----------------
    olivetti = _load_olivetti_embeddings()
    cross = np.asarray(fem_metrics.pairwise_distances(olivetti, scan_means), dtype=np.float64)
    per_olivetti_min = cross.min(axis=1)
    frame_level_cross = np.asarray(
        fem_metrics.pairwise_distances(olivetti, stack.reshape(-1, 512)), dtype=np.float64
    )
    cross_domain = {
        "olivetti_min_distance_to_any_live_scan": float(per_olivetti_min.min()),
        "olivetti_mean_of_min": float(per_olivetti_min.mean()),
        "nearest_live_scan": names[int(np.argmin(cross)) % N_SCANS],
        "n_olivetti": int(olivetti.shape[0]),
        "frame_level_min_distance": float(frame_level_cross.min()),
        "caveat": CROSS_DOMAIN_CAVEAT,
    }
    print(f"[live] cross-domain: nearest Olivetti->my-scan distance {cross_domain['olivetti_min_distance_to_any_live_scan']:.4f} "
          f"(mean of per-Olivetti minima {cross_domain['olivetti_mean_of_min']:.4f})")

    # ---------------- 8. frame temporal structure (limitation bound) ----------------
    lags, lag_means, lag_counts = [], [], []
    for lag in range(1, N_FRAMES):
        values = np.asarray(
            [np.linalg.norm(stack[k][i] - stack[k][i + lag]) for k in range(N_SCANS)
             for i in range(N_FRAMES - lag)],
            dtype=np.float64,
        )
        lags.append(lag)
        lag_means.append(float(values.mean()))
        lag_counts.append(int(values.size))
    frame_lag = {
        "lags": lags,
        "mean_distance": lag_means,
        "n_pairs": lag_counts,
        "interpretation": "distance grows with frame lag, so frames within a scan are not "
                          "independent draws (slow head/pose drift during the scan); the "
                          "effective sample size is below 10.",
    }
    frame_outlier = {"frame_index": list(range(N_FRAMES)), "mean_distance": []}
    for fi in range(N_FRAMES):
        values = []
        for k in range(N_SCANS):
            others = np.delete(stack[k], fi, axis=0)
            values.extend(np.linalg.norm(stack[k][fi] - others, axis=1).tolist())
        frame_outlier["mean_distance"].append(float(np.mean(values)))
    print(f"[live] frame lag 1->9: {lag_means[0]:.3f} -> {lag_means[-1]:.3f}; "
          f"frame-0 mean distance {frame_outlier['mean_distance'][0]:.3f} "
          f"(next highest {max(frame_outlier['mean_distance'][1:]):.3f})")

    # ---------------- JSON payload (DESIGN.md §6 schema + requested extras) ----------------
    notes = [
        "Distances are Euclidean between unit vectors; cosine equivalents are "
        "cos = 1 - d^2/2 and are listed under cosine_equivalents.",
        "Frames within a scan are temporally correlated (frame_lag): the 10 frames are "
        "not 10 independent samples, so effective sample size < 10.",
        "Scan means are renormalised before use; mean-vector norms before renormalisation "
        f"were {raw_mean_norms.min():.4f}..{raw_mean_norms.max():.4f} "
        f"(mean {raw_mean_norms.mean():.4f}), so pooling barely changes direction.",
        "The cross-domain check is domain-confounded by capture, crop and resolution; it "
        "is not a calibrated impostor distribution and is neither an upper nor a lower "
        "bound on webcam-domain stranger distance.",
        "design is unbalanced by necessity: room light has front+left only; flash has "
        "front/left/right/bottom; two repeats per condition; one subject.",
    ]
    payload = {
        "common": common_header(device, notes=notes),
        "design": {
            "subject": scan_set.subject,
            "scans": N_SCANS,
            "frames_per_scan": N_FRAMES,
            "captured": scan_set.captured,
            "conditions": CONDITION_ORDER,
            "imbalance_note": "room light has front+left only; flash has front/left/right/bottom; "
                              "two repeats per condition; one subject",
        },
        "within_scan_jitter": within,
        "same_condition": same_condition,
        "noise_floor_mean": noise_floor_mean,
        "intensity_effect": intensity_effect,
        "direction_effect": direction_effect,
        "direction_to_noise_ratio": direction_to_noise_ratio,
        "pooling": pooling,
        "worst_genuine_distance": worst,
        "cross_domain_impostor": cross_domain,
        "matrix_names": names,
        "matrix_mean_distance": matrix.tolist(),
        "frame_lag": frame_lag,
        "frame_outlier": frame_outlier,
        "cosine_equivalents": {
            "noise_floor_mean": _cos_block(noise_floor_mean),
            "intensity_effect_mean": _cos_block(float(intensity_effect["mean"])),
            "direction_effect_mean": _cos_block(direction_effect_mean),
            "worst_genuine_frame": _cos_block(worst_value),
            "worst_genuine_scan_mean": _cos_block(worst_scan_mean),
            "cross_domain_min": _cos_block(cross_domain["olivetti_min_distance_to_any_live_scan"]),
        },
    }

    # ---------------- figures ----------------
    print("[live] figures")
    tick_labels = [f"{s.condition} r{s.repeat}" for s in ordered]
    save_heatmap(
        matrix,
        tick_labels,
        FIGURES_DIR / "live_heatmap",
        title="Mean embedding distance between scans (one subject; darker = further)",
    )
    save_lighting_panel(payload, FIGURES_DIR / "live_lighting")
    save_pooling(frame_level, pooled_values, FIGURES_DIR / "live_pooling")

    out_path = RESULTS_DIR / "live.json"
    save_json(payload, out_path)
    print(f"[live] wrote {out_path} in {time.time() - started:.1f}s")
    return payload


if __name__ == "__main__":
    main()
