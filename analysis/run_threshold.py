"""E4 — open-set verification / the FaceID threshold, Map A vs Map B.

Why this script exists
----------------------
Stage 5 of the original plan is the security-convenience dial: pick a distance
threshold, and count what leaks in (false accepts) against what gets locked out
(false rejects).  This script runs that as a controlled verification protocol on
Olivetti, where ground truth exists, and does it identically for raw pixels and for
the frozen deep embeddings.  This is the comparison the plan calls the potential
headline result.

Protocol (DESIGN.md §5 E4), identical for both representations:
  * per identity: enrol 5 images, probe the other 5 (genuine trials);
  * impostor trials: for each identity, every one of the 10 images of each of the
    other 39 identities is scored against that identity's enrolment (390 probes
    per identity, 15 600 impostor trials total);
  * score = minimum Euclidean distance from the probe to the 5 enrolled images
    (verification against the enrolled *set*, as in the original plan's "distance
    from the new vector to each saved one"); accept if score <= threshold;
  * primary split selection: 5 random 5/5 splits seeded 0-4, averaged;
  * secondary: fixed first-half / second-half split, reported only as a
    session-bias check, never as the headline.

Scores are FULL-dimensional: 4096-d raw pixels and 512-d unit embeddings.  No PCA
is involved here, deliberately — measuring verification on a 2-D plot would be
measuring plotting settings (plan trap #3), which is why this runner uses
fem.metrics and not fem.maps.

Run: ``$PY analysis/run_threshold.py``
Outputs: results/threshold.json, results/threshold_curves.npz,
         figures/threshold_curves.{png,pdf}
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

from analysis._common import (  # noqa: E402
    FIGURES_DIR,
    MAP_COLORS,
    RESULTS_DIR,
    SEED,
    common_header,
    describe_device,
    save_json,
)

N_IDENTITIES = 40
N_IMAGES = 400
IMAGES_PER_IDENTITY = 10
ENROL = 5
PROBE = 5
SPLIT_SEEDS = [0, 1, 2, 3, 4]
IMPOSTOR_PROBES_PER_IDENTITY = (N_IDENTITIES - 1) * IMAGES_PER_IDENTITY  # 390
FRR_CONVENIENCE = 0.01


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _split(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """5/5 split per identity, seeded and deterministic.

    Returns (enrol_indices, probe_indices) each (40,5).  Randomness is per identity
    over its own 10 images, in identity order, from one seeded generator.
    """
    rng = np.random.default_rng(seed)
    enrol = np.empty((N_IDENTITIES, ENROL), dtype=int)
    probe = np.empty((N_IDENTITIES, PROBE), dtype=int)
    for identity in range(N_IDENTITIES):
        idx = np.arange(identity * IMAGES_PER_IDENTITY, (identity + 1) * IMAGES_PER_IDENTITY)
        perm = rng.permutation(idx)
        enrol[identity] = psorted(perm[:ENROL])
        probe[identity] = psorted(perm[ENROL:])
    return enrol, probe


def psorted(values: np.ndarray) -> np.ndarray:
    """Sorted copy: order inside the enrol/probe set must not matter."""
    return np.sort(np.asarray(values, dtype=int))


def fixed_split() -> tuple[np.ndarray, np.ndarray]:
    """Secondary check: first half of each identity enrols, second half probes."""
    enrol = np.empty((N_IDENTITIES, ENROL), dtype=int)
    probe = np.empty((N_IDENTITIES, PROBE), dtype=int)
    for identity in range(N_IDENTITIES):
        base = identity * IMAGES_PER_IDENTITY
        enrol[identity] = np.arange(base, base + ENROL)
        probe[identity] = np.arange(base + ENROL, base + IMAGES_PER_IDENTITY)
    return enrol, probe


def _scores(
    distances: np.ndarray, enrol: np.ndarray, probe: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Genuine and impostor scores for one split, from the full 400x400 distance matrix.

    Genuine: each probe image scored against its own identity's enrolled set.
    Impostor: for each identity, all images of the other 39 identities scored
    against that identity's enrolled set (390 per identity).
    """
    genuine = []
    impostor = []
    for identity in range(N_IDENTITIES):
        own_probe = probe[identity]
        genuine.append(distances[np.ix_(own_probe, enrol[identity])].min(axis=1))
        other_probe = np.concatenate(
            [probe[j] for j in range(N_IDENTITIES) if j != identity]
            + [enrol[j] for j in range(N_IDENTITIES) if j != identity]
        )
        impostor.append(distances[np.ix_(other_probe, enrol[identity])].min(axis=1))
    genuine = np.concatenate(genuine).astype(np.float64)
    impostor = np.concatenate(impostor).astype(np.float64)
    assert genuine.size == N_IDENTITIES * PROBE, f"genuine trials {genuine.size}"
    assert impostor.size == N_IDENTITIES * IMPOSTOR_PROBES_PER_IDENTITY, (
        f"impostor trials {impostor.size}"
    )
    return genuine, impostor


def _mean_template_scores(
    X: np.ndarray, enrol: np.ndarray, probe: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Alternative score rule: distance to the mean enrolment template.

    Used only as a sensitivity check on the primary "min distance to the
    enrolled set" rule.  Templates are unnormalised means: renormalising a
    4096-d raw-pixel mean is meaningless (its norm is ~40, not 1), so the same
    rule is applied to both maps.
    """
    genuine = []
    impostor = []
    for identity in range(N_IDENTITIES):
        template = X[enrol[identity]].mean(axis=0)
        genuine.append(np.linalg.norm(X[probe[identity]] - template, axis=1))
        others = np.concatenate(
            [probe[j] for j in range(N_IDENTITIES) if j != identity]
            + [enrol[j] for j in range(N_IDENTITIES) if j != identity]
        )
        impostor.append(np.linalg.norm(X[others] - template, axis=1))
    return (
        np.concatenate(genuine).astype(np.float64),
        np.concatenate(impostor).astype(np.float64),
    )


def _operating_metrics(genuine: np.ndarray, impostor: np.ndarray, fem_metrics) -> dict:
    """All per-split verification numbers, with distances as scores (lower = genuine)."""
    auc = float(fem_metrics.roc_auc(genuine, impostor))
    # fem.metrics takes genuine/impostor distance arrays; a sensible implementation
    # returns AUC > 0.5 for a useful representation. Fail loudly if inverted.
    assert auc > 0.5, f"AUC {auc:.3f} <= 0.5: score direction looks inverted"
    eer_value, eer_threshold = fem_metrics.eer(genuine, impostor)
    eer_threshold = float(eer_threshold)

    far_at_eer = float(np.mean(impostor <= eer_threshold))
    frr_at_eer = float(np.mean(genuine > eer_threshold))

    # Convenience operating point: threshold set so 1% of genuine probes are rejected.
    threshold_frr_1pct = float(np.quantile(genuine, 1.0 - FRR_CONVENIENCE))
    far_at_frr_1pct = float(np.mean(impostor <= threshold_frr_1pct))
    frr_at_frr_1pct = float(np.mean(genuine > threshold_frr_1pct))

    return {
        "eer": float(eer_value),
        "eer_threshold": eer_threshold,
        "tar_at_far_1e-3": float(fem_metrics.tar_at_far(genuine, impostor, 1e-3)),
        "tar_at_far_1e-2": float(fem_metrics.tar_at_far(genuine, impostor, 1e-2)),
        "auc": auc,
        "far_at_frr_1pct": far_at_frr_1pct,
        "threshold_at_frr_1pct": threshold_frr_1pct,
        "far_at_eer": far_at_eer,
        "frr_at_eer": frr_at_eer,
        "frr_at_frr_1pct": frr_at_frr_1pct,
    }


def _aggregate(per_split: list[dict]) -> tuple[dict, dict]:
    """Mean and std across splits for every numeric key."""
    keys = [k for k in per_split[0] if isinstance(per_split[0][k], float)]
    mean = {k: float(np.mean([s[k] for s in per_split])) for k in keys}
    std = {k: float(np.std([s[k] for s in per_split], ddof=1)) for k in keys}
    return mean, std


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> dict:
    started = time.time()
    device = describe_device()

    from fem import metrics as fem_metrics
    from fem.data import load_olivetti

    # ---------------- representations (full-dimensional) ----------------
    images, labels = load_olivetti()
    labels = np.asarray(labels, dtype=int)
    assert images.shape == (N_IMAGES, 64, 64)
    X_a = np.asarray(images, dtype=np.float64).reshape(N_IMAGES, -1)

    cache = RESULTS_DIR / "embeddings_olivetti.npz"
    if not cache.exists():
        raise FileNotFoundError(f"{cache} missing: run analysis/run_olivetti.py first (E2 caches it)")
    with np.load(cache) as npz:
        candidates = {
            key: npz[key] for key in npz.files
            if npz[key].shape == (N_IMAGES, 512) and np.issubdtype(npz[key].dtype, np.number)
        }
    if not candidates:
        raise KeyError(f"no (400,512) array in {cache}")
    key = "embeddings" if "embeddings" in candidates else sorted(candidates)[0]
    X_b = np.asarray(candidates[key], dtype=np.float64)
    norms = np.linalg.norm(X_b, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), (
        f"Map B embeddings are not unit-norm: {norms.min():.6f}..{norms.max():.6f}"
    )

    print("[threshold] precomputing full 400x400 distance matrices")
    distances_a = np.asarray(fem_metrics.pairwise_distances(X_a), dtype=np.float64)
    distances_b = np.asarray(fem_metrics.pairwise_distances(X_b), dtype=np.float64)
    assert distances_a.shape == (N_IMAGES, N_IMAGES)
    assert np.allclose(np.diag(distances_a), 0.0, atol=1e-6), "distance to self is not zero"

    # ---------------- primary: 5 seeded random splits ----------------
    results = {"map_a": [], "map_b": []}
    pooled_scores = {"map_a": [], "map_b": []}
    split_sweeps = {"map_a": [], "map_b": []}
    for split_seed in SPLIT_SEEDS:
        enrol, probe = _split(split_seed)
        for map_name, distances in (("map_a", distances_a), ("map_b", distances_b)):
            genuine, impostor = _scores(distances, enrol, probe)
            metrics = _operating_metrics(genuine, impostor, fem_metrics)
            metrics["seed"] = split_seed
            results[map_name].append(metrics)
            pooled_scores[map_name].append((genuine, impostor))
            # Per-split FAR/FRR curve, kept in the npz artefact below.
            split_sweeps[map_name].append(
                {
                    key: np.asarray(value, dtype=np.float64)
                    for key, value in fem_metrics.threshold_sweep(genuine, impostor).items()
                }
            )
        print(
            f"[threshold] split seed {split_seed}: "
            f"Map A EER {results['map_a'][-1]['eer']:.4f}, "
            f"Map B EER {results['map_b'][-1]['eer']:.4f}"
        )

    map_blocks = {}
    for map_name in ("map_a", "map_b"):
        mean, std = _aggregate(results[map_name])
        map_blocks[map_name] = {"per_split": results[map_name], "mean": mean, "std": std}
        print(
            f"[threshold] {map_name}: EER {mean['eer']:.4f} ± {std['eer']:.4f}, "
            f"AUC {mean['auc']:.5f}, TAR@FAR1e-3 {mean['tar_at_far_1e-3']:.3f}, "
            f"TAR@FAR1e-2 {mean['tar_at_far_1e-2']:.3f}, "
            f"FAR@FRR1% {mean['far_at_frr_1pct']:.3%}"
        )

    # ---------------- score-rule sensitivity (mean-template alternative) ----------------
    # Is the result an artefact of "min distance to the enrolled set"?  The other
    # standard rule is "distance to the mean enrolment template".  Averaging a
    # 4096-d raw-pixel vector is still meaningful; renormalising it is not (its
    # norm is ~40, not 1), so neither map is renormalised here.
    sensitivity = {}
    for map_name, X in (("map_a", X_a), ("map_b", X_b)):
        eers, aucs = [], []
        for split_seed in SPLIT_SEEDS:
            enrol, probe = _split(split_seed)
            genuine, impostor = _mean_template_scores(X, enrol, probe)
            eers.append(float(fem_metrics.eer(genuine, impostor)[0]))
            aucs.append(float(fem_metrics.roc_auc(genuine, impostor)))
        sensitivity[map_name] = {
            "eer_mean": float(np.mean(eers)),
            "eer_std": float(np.std(eers, ddof=1)),
            "auc_mean": float(np.mean(aucs)),
        }
    print(
        f"[threshold] score-rule sensitivity (mean template): "
        f"Map A EER {sensitivity['map_a']['eer_mean']:.4f}, "
        f"Map B EER {sensitivity['map_b']['eer_mean']:.4f}"
    )

    # ---------------- secondary: fixed first/second-half split ----------------
    enrol_f, probe_f = fixed_split()
    session = {}
    for map_name, distances in (("map_a", distances_a), ("map_b", distances_b)):
        genuine, impostor = _scores(distances, enrol_f, probe_f)
        session[map_name] = _operating_metrics(genuine, impostor, fem_metrics)
    assert labels.tolist() == np.repeat(np.arange(N_IDENTITIES), IMAGES_PER_IDENTITY).tolist()
    session_bias_check = {
        "enrol_indices": [0, 1, 2, 3, 4],
        "probe_indices": [5, 6, 7, 8, 9],
        "map_a": session["map_a"],
        "map_b": session["map_b"],
        "note": "fixed first-half/second-half split per identity; secondary only, not the "
                "paper headline (it removes all split randomness but can encode capture order)",
    }

    # ---------------- headline sentence (derived, not typed) ----------------
    a_mean, b_mean = map_blocks["map_a"]["mean"], map_blocks["map_b"]["mean"]
    a_tar, b_tar = a_mean["tar_at_far_1e-2"], b_mean["tar_at_far_1e-2"]
    a_far, b_far = a_mean["far_at_frr_1pct"], b_mean["far_at_frr_1pct"]
    if b_far > 0:
        far_ratio = a_far / b_far
        far_gap = f"at FRR=1%: FAR {a_far:.2%} (Map A) vs {b_far:.3%} (Map B), {far_ratio:.1f}x more impostors accepted by Map A"
    else:
        far_gap = f"at FRR=1%: FAR {a_far:.2%} (Map A) vs 0% (Map B)"
    statement = (
        f"At a 1% false-accept rate, Map B accepts {b_tar:.1%} of genuine probes versus "
        f"{a_tar:.1%} for Map A; at matched 1% false-reject rate Map A accepts "
        f"{a_far:.2%} of impostors versus {b_far:.3%} for Map B."
    )
    headline = {
        "statement": statement,
        "map_b_eer": b_mean["eer"],
        "map_a_eer": a_mean["eer"],
        "eer_reduction_factor": float(a_mean["eer"] / b_mean["eer"]) if b_mean["eer"] > 0 else None,
        "far_gap_at_matched_frr": far_gap,
    }

    # ---------------- figure from the pooled trials ----------------
    # Two panels, because raw-pixel distances (~10) and embedding distances (~1)
    # differ by an order of magnitude: on a shared axis one map's curves collapse
    # onto the y-axis and the figure hides the comparison it exists to show.
    curves = {}
    npz_payload = {}
    for map_name, label in (("map_a", "Map A"), ("map_b", "Map B")):
        genuine_all = np.concatenate([g for g, _ in pooled_scores[map_name]])
        impostor_all = np.concatenate([i for _, i in pooled_scores[map_name]])
        curves[label] = (genuine_all, impostor_all)
        sweep = fem_metrics.threshold_sweep(genuine_all, impostor_all)
        for key, value in sweep.items():
            npz_payload[f"{map_name}_pooled_{key}"] = np.asarray(value, dtype=np.float64)
        for index, split_sweep in enumerate(split_sweeps[map_name]):
            for key, value in split_sweep.items():
                npz_payload[f"{map_name}_split{index}_{key}"] = value
    np.savez(RESULTS_DIR / "threshold_curves.npz", **npz_payload)
    from fem.plotting import save_threshold_sweep

    save_threshold_sweep(
        curves, FIGURES_DIR / "threshold_curves", separate_panels=True
    )

    # ---------------- JSON payload (DESIGN.md §6 schema) ----------------
    notes = [
        "Score = minimum Euclidean distance from a probe image to the 5 enrolment "
        "images; decision = accept if score <= threshold. Trials are full-dimensional, "
        "no PCA anywhere in this runner.",
        "Impostor trials: for each identity, the 390 images of the other 39 identities "
        "are each scored against that identity's enrolment.",
        "threshold_curves figure is drawn from the genuine/impostor scores pooled over "
        "the 5 primary splits.",
        "Trial granularity: 200 genuine trials per split give an FRR resolution of 0.5%; "
        "15,600 impostor trials give a FAR resolution of 0.0064%. Map B's mean EER "
        "(0.22%) is below the FRR grid resolution, so it should be read as 'at or below "
        "the measurement floor', not as a precise 0.22%.",
    ]
    payload = {
        "common": common_header(device, notes=notes),
        "protocol": {
            "enrol": ENROL,
            "probe": PROBE,
            "impostor_probes_per_identity": IMPOSTOR_PROBES_PER_IDENTITY,
            "genuine_trials_per_split": N_IDENTITIES * PROBE,
            "impostor_trials_per_split": N_IDENTITIES * IMPOSTOR_PROBES_PER_IDENTITY,
            "splits": "5 random 5/5 splits, seeds 0-4",
            "identical_for": ["map_a", "map_b"],
            "score": "minimum Euclidean distance from probe to the 5 enrolment images",
            "accept_rule": "distance <= threshold",
        },
        "map_a": map_blocks["map_a"],
        "map_b": map_blocks["map_b"],
        "headline": headline,
        "session_bias_check": session_bias_check,
        "score_rule_sensitivity": {
            "mean_template": sensitivity,
            "note": "Primary rule is the minimum distance to the 5 enrolment images. "
            "Averaging the enrolment images into one template gives the same ranking, "
            "so the Map A vs Map B conclusion is not an artefact of the score rule.",
        },
    }
    out_path = RESULTS_DIR / "threshold.json"
    save_json(payload, out_path)
    print(f"[threshold] wrote {out_path} in {time.time() - started:.1f}s")
    print(f"[threshold] HEADLINE: {statement}")
    return payload


if __name__ == "__main__":
    main()
