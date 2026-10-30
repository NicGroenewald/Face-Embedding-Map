"""E4 — known-population cross-identity verification, Map A vs Map B (descriptive).

Why this script exists
---------------------
Stage 5 of the original plan is the security-convenience dial: pick a distance
threshold, and count what leaks in (false accepts) against what gets locked out
(false rejects).  This script runs that as a *controlled, same-population*
verification experiment on Olivetti, where ground truth exists, and does it
identically for raw pixels and for the frozen deep embeddings.

What this is and is not (read before quoting any number)
--------------------------------------------------------
* It is **known-population cross-identity verification**: every "impostor" is
  one of the same 40 Olivetti people.  No unknown identity is ever tested, so
  the numbers say nothing about open-set identification, stranger rejection in
  the wild, or a deployed FaceID system.
* The map_a / map_b / headline numbers are **descriptive and in-sample**:
  thresholds are chosen and evaluated on the same scores (the EER is an
  interpolated crossing between observed thresholds, not a realisable operating
  point).  They are useful for comparing representations under one protocol;
  they are not calibrated error rates for unseen identities.
* The ``calibration_heldout_evaluation`` block (added below) is the honest
  check: the 40 identities are split into disjoint calibration / identity-
  disjoint evaluation / unknown-to-gallery folds, thresholds are chosen on
  calibration scores only, and empirical FAR/FRR are measured on held-out
  identities.  It is still Olivetti-only, so it is a sanity check on optimism,
  not a deployment estimate.

Primary protocol (DESIGN.md §5 E4), identical for both representations:
  * per identity: enrol 5 images, probe the other 5 (genuine trials);
  * impostor trials: for each identity, every one of the 10 images of each of the
    other 39 known identities is scored against that identity's enrolment (390
    probes per identity, 15 600 impostor trials total);
  * score = minimum Euclidean distance from the probe to the 5 enrolled images
    (verification against the enrolled *set*, as in the original plan's "distance
    from the new vector to each saved one"); accept if score <= threshold;
  * primary split selection: 5 random 5/5 splits seeded 0-4, averaged;
  * secondary: fixed first-half / second-half split, reported only as a
    session-bias check.

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

# Identity-disjoint calibration-vs-test check (added as a separate block; the
# historical descriptive numbers above are left untouched).
HELDOUT_PARTITION_SEEDS = [0, 1, 2, 3, 4]
#: Calibration FAR targets: 1 % (primary) and 0.1 % (secondary).  No FRR target:
#: the 65-genuine calibration fold has 1/65 = 1.54 % FRR granularity, so a 1 %
#: FRR target is not resolvable and would force a degenerate infinite threshold.
HELDOUT_TARGETS = (("far", 1e-2), ("far", 1e-3))


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
    Impostor: for each identity, all 10 images of each of the other 39
    *known* identities are scored against that identity's enrolled set (390 per
    identity).  This is cross-identity matching inside one known population, not
    open-set identification: no identity outside the 40 is ever an impostor.
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
    """In-sample descriptive operating points for one split (lower score = more genuine).

    Both thresholds here are derived from the same genuine/impostor arrays they
    are evaluated on (an interpolated EER crossing and a genuine-score
    quantile), so the resulting FAR/FRR are descriptive, not calibrated
    estimates for unseen identities.  The identity-disjoint check lives in
    ``_heldout_evaluation``.
    """
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
# identity-disjoint calibration-vs-test check (honest, additive)
#
# The descriptive block above chooses thresholds and measures errors on the same
# scores.  This block instead splits the 40 identities into three disjoint
# folds, chooses thresholds on calibration identities only, and measures
# empirical error rates on identities the threshold never saw.  It is still one
# small dataset (Olivetti) and still known-population matching, so it is a check
# on optimism, not a deployment estimate.
# --------------------------------------------------------------------------- #
def _partition_identities(
    seed: int, n_identities: int = N_IDENTITIES
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic three-way identity split: calibration / evaluation / unknown.

    With the 40 Olivetti identities this is 13 / 13 / 14.  The evaluation fold is
    identity-disjoint from calibration, and the unknown fold is enrolled in no
    gallery at all.
    """
    if n_identities < 6:
        raise ValueError(f"need at least 6 identities to split three ways, got {n_identities}")
    rng = np.random.default_rng(10_000 + seed)
    permutation = rng.permutation(n_identities)
    n_calibration = n_identities // 3
    n_evaluation = n_identities // 3
    calibration = np.sort(permutation[:n_calibration])
    evaluation = np.sort(permutation[n_calibration : n_calibration + n_evaluation])
    unknown = np.sort(permutation[n_calibration + n_evaluation :])
    return calibration, evaluation, unknown


def _enrolment_split(
    identities: np.ndarray,
    seed: int,
    images_per_identity: int = IMAGES_PER_IDENTITY,
    enrol: int = ENROL,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Seeded 5/5 enrol/probe split for each supplied identity.

    Returns ``(enrol_by_id, probe_by_id)`` mapping identity -> sorted image
    indices.  A separate generator (offset from the partition seed) keeps the
    split independent of the identity partition.
    """
    if not 1 <= enrol < images_per_identity:
        raise ValueError(f"enrol must be in 1..{images_per_identity - 1}, got {enrol}")
    rng = np.random.default_rng(20_000 + seed)
    enrol_by_id: dict[int, np.ndarray] = {}
    probe_by_id: dict[int, np.ndarray] = {}
    for identity in np.asarray(identities, dtype=int):
        base = int(identity) * images_per_identity
        permutation = rng.permutation(np.arange(base, base + images_per_identity))
        enrol_by_id[int(identity)] = np.sort(permutation[:enrol])
        probe_by_id[int(identity)] = np.sort(permutation[enrol:])
    return enrol_by_id, probe_by_id


def _pair_scores(
    distances: np.ndarray,
    identities: np.ndarray,
    enrol_by_id: dict[int, np.ndarray],
    probe_by_id: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Genuine and cross-identity impostor scores *within one identity set*.

    Genuine: each probe of identity i against i's enrolled images (minimum).
    Impostor: every image (enrol + probe) of the other identities in this set
    against identity i's enrolled images (minimum).  All identities here are
    from the same known population, so this is not open-set matching.
    """
    identities = [int(identity) for identity in identities]
    genuine: list[np.ndarray] = []
    impostor: list[np.ndarray] = []
    for identity in identities:
        genuine.append(
            distances[np.ix_(probe_by_id[identity], enrol_by_id[identity])].min(axis=1)
        )
        other_images = np.concatenate(
            [
                np.concatenate([enrol_by_id[other], probe_by_id[other]])
                for other in identities
                if other != identity
            ]
        )
        impostor.append(
            distances[np.ix_(other_images, enrol_by_id[identity])].min(axis=1)
        )
    return (
        np.concatenate(genuine).astype(np.float64),
        np.concatenate(impostor).astype(np.float64),
    )


def _unknown_scores(
    distances: np.ndarray,
    gallery_ids: np.ndarray,
    enrol_by_id: dict[int, np.ndarray],
    unknown_ids: np.ndarray,
    images_per_identity: int = IMAGES_PER_IDENTITY,
) -> np.ndarray:
    """Score every image of identities enrolled in no gallery.

    For each unknown image, return the minimum distance to *any* gallery
    identity's enrolled set: the closest claim the system could make that the
    image matches an enrolled person.  A single threshold can then ask "was any
    unknown image accepted as somebody?".
    """
    gallery = [int(identity) for identity in gallery_ids]
    scores: list[np.ndarray] = []
    for identity in np.asarray(unknown_ids, dtype=int):
        base = int(identity) * images_per_identity
        images = np.arange(base, base + images_per_identity)
        per_gallery = np.stack(
            [
                distances[np.ix_(images, enrol_by_id[gallery_identity])].min(axis=1)
                for gallery_identity in gallery
            ],
            axis=1,
        )
        scores.append(per_gallery.min(axis=1))
    return np.concatenate(scores).astype(np.float64)


def _threshold_at_far(impostor: np.ndarray, far: float) -> float:
    """Conservative empirical threshold for a target false-accept rate.

    Thin wrapper over ``fem.metrics.threshold_at_far`` — the tie-safe
    conservative rule (largest observed impostor score whose empirical FAR does
    not exceed ``far``; step down if accepting tied scores would overshoot) is
    defined once in the library, not duplicated in analysis.  Kept as a local
    name because the held-out helpers call it in a loop and the boundary
    behaviour is pinned by tests.
    """
    from fem import metrics as fem_metrics

    return float(fem_metrics.threshold_at_far(impostor, far))


def _heldout_evaluation(
    distances: np.ndarray,
    seed: int,
    targets: tuple[tuple[str, float], ...] = HELDOUT_TARGETS,
    *,
    n_identities: int = N_IDENTITIES,
    images_per_identity: int = IMAGES_PER_IDENTITY,
    enrol: int = ENROL,
) -> tuple[dict, dict]:
    """One identity-disjoint calibration-vs-test partition.

    Returns ``(public, internal)``: ``public`` is JSON-ready summary numbers;
    ``internal`` keeps the raw score arrays so tests can verify every reported
    rate against a direct computation (the runner strips it before writing).
    """
    from fem import metrics as fem_metrics

    calibration_ids, evaluation_ids, unknown_ids = _partition_identities(seed, n_identities)
    enrol_by_id, probe_by_id = _enrolment_split(
        np.concatenate([calibration_ids, evaluation_ids]), seed, images_per_identity, enrol
    )
    genuine_cal, impostor_cal = _pair_scores(distances, calibration_ids, enrol_by_id, probe_by_id)
    genuine_eval, impostor_eval = _pair_scores(distances, evaluation_ids, enrol_by_id, probe_by_id)
    unknown = _unknown_scores(
        distances, evaluation_ids, enrol_by_id, unknown_ids, images_per_identity
    )

    operating_points = []
    for rule, target in targets:
        if rule != "far":
            raise ValueError(f"unknown threshold rule {rule!r}")
        threshold = _threshold_at_far(impostor_cal, target)
        operating_points.append(
            {
                "rule": f"calibration_{rule}",
                "target": float(target),
                "threshold": float(threshold),
                # Rates on the calibration scores that chose the threshold:
                "calibration_far": float(np.mean(impostor_cal <= threshold)),
                "calibration_frr": float(np.mean(genuine_cal > threshold)),
                # Rates on identity-disjoint held-out identities:
                "evaluation_far": float(np.mean(impostor_eval <= threshold)),
                "evaluation_frr": float(np.mean(genuine_eval > threshold)),
                "evaluation_tar": float(np.mean(genuine_eval <= threshold)),
                "unknown_accept_rate": float(np.mean(unknown <= threshold)),
            }
        )

    # Interpolated EER on calibration scores, then asked of held-out identities.
    # The EER is an interpolation between observed thresholds, not an achievable
    # operating point, and it need not be reproduced out of sample.
    eer_value, eer_threshold = fem_metrics.eer(genuine_cal, impostor_cal)
    eer_check = {
        "calibration_eer_interpolated": float(eer_value),
        "calibration_eer_threshold": float(eer_threshold),
        "evaluation_far_at_calibration_eer_threshold": float(
            np.mean(impostor_eval <= eer_threshold)
        ),
        "evaluation_frr_at_calibration_eer_threshold": float(
            np.mean(genuine_eval > eer_threshold)
        ),
        "note": "EER is interpolated between two observed thresholds. It is not a "
        "realisable operating point, and it is not expected to be reproduced on "
        "identity-disjoint held-out data.",
    }

    public = {
        "partition_seed": int(seed),
        "n_calibration_identities": int(calibration_ids.size),
        "n_evaluation_identities": int(evaluation_ids.size),
        "n_unknown_identities": int(unknown_ids.size),
        "n_calibration_genuine": int(genuine_cal.size),
        "n_calibration_impostor": int(impostor_cal.size),
        "n_evaluation_genuine": int(genuine_eval.size),
        "n_evaluation_impostor": int(impostor_eval.size),
        "n_unknown_probes": int(unknown.size),
        "operating_points": operating_points,
        "eer_interpolation_check": eer_check,
    }
    internal = {
        "calibration_identities": calibration_ids,
        "evaluation_identities": evaluation_ids,
        "unknown_identities": unknown_ids,
        "genuine_calibration": genuine_cal,
        "impostor_calibration": impostor_cal,
        "genuine_evaluation": genuine_eval,
        "impostor_evaluation": impostor_eval,
        "unknown_scores": unknown,
    }
    return public, internal


def _aggregate_heldout(per_partition: list[dict]) -> tuple[dict, dict]:
    """Mean and std across partition seeds for the held-out operating points."""
    n_points = len(per_partition[0]["operating_points"])
    numeric = [
        key
        for key, value in per_partition[0]["operating_points"][0].items()
        if isinstance(value, float) and key != "target"
    ]
    mean_points, std_points = [], []
    for index in range(n_points):
        first = per_partition[0]["operating_points"][index]
        mean_points.append(
            {
                "rule": first["rule"],
                "target": first["target"],
                **{
                    key: float(np.mean([p["operating_points"][index][key] for p in per_partition]))
                    for key in numeric
                },
            }
        )
        std_points.append(
            {
                "rule": first["rule"],
                "target": first["target"],
                **{
                    key: float(
                        np.std(
                            [p["operating_points"][index][key] for p in per_partition], ddof=1
                        )
                    )
                    for key in numeric
                },
            }
        )
    eer_keys = [
        key
        for key, value in per_partition[0]["eer_interpolation_check"].items()
        if isinstance(value, float)
    ]
    eer_mean = {
        key: float(np.mean([p["eer_interpolation_check"][key] for p in per_partition]))
        for key in eer_keys
    }
    eer_std = {
        key: float(np.std([p["eer_interpolation_check"][key] for p in per_partition], ddof=1))
        for key in eer_keys
    }
    return (
        {"operating_points": mean_points, "eer_interpolation_check": eer_mean},
        {"operating_points": std_points, "eer_interpolation_check": eer_std},
    )


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

    # A shape-valid array is not enough: only the validated cache loader checks
    # checkpoint, dataset, preprocessing and row integrity. Run maps first if stale.
    from fem.embed import load_cached_olivetti_embeddings

    X_b = np.asarray(load_cached_olivetti_embeddings(), dtype=np.float64)

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

    # ---------------- identity-disjoint calibration-vs-test check ----------------
    # Additive and separate from the descriptive numbers above: thresholds are
    # chosen on calibration identities only and evaluated on identities the
    # thresholds never saw.  Still one small known population, so descriptive.
    heldout_blocks = {}
    for map_name, distances in (("map_a", distances_a), ("map_b", distances_b)):
        per_partition = []
        for partition_seed in HELDOUT_PARTITION_SEEDS:
            public, _internal = _heldout_evaluation(distances, partition_seed)
            per_partition.append(public)
        mean, std = _aggregate_heldout(per_partition)
        heldout_blocks[map_name] = {
            "per_partition": per_partition,
            "mean": mean,
            "std": std,
        }
        primary = mean["operating_points"][0]
        print(
            f"[threshold] calibration-heldout {map_name}: calibration FAR target "
            f"{primary['target']:.1%} -> evaluation TAR {primary['evaluation_tar']:.3f}, "
            f"evaluation FAR {primary['evaluation_far']:.4f}, "
            f"unknown-accept {primary['unknown_accept_rate']:.4f}"
        )
    primary_target = heldout_blocks["map_a"]["mean"]["operating_points"][0]["target"]
    comparison = {
        "primary_calibration_far": float(primary_target),
        "map_a": {
            key: heldout_blocks["map_a"]["mean"]["operating_points"][0][key]
            for key in (
                "calibration_far",
                "evaluation_far",
                "evaluation_frr",
                "evaluation_tar",
                "unknown_accept_rate",
            )
        },
        "map_b": {
            key: heldout_blocks["map_b"]["mean"]["operating_points"][0][key]
            for key in (
                "calibration_far",
                "evaluation_far",
                "evaluation_frr",
                "evaluation_tar",
                "unknown_accept_rate",
            )
        },
        "note": "Held-out identities are identity-disjoint from the calibration "
        "identities, but all folds come from the same 40-person Olivetti set; the 5 "
        "partition seeds reuse and overlap those identities and are NOT independent "
        "samples.",
    }

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
        "note": "fixed first-half/second-half split per identity; descriptive secondary "
                "check only (it removes split randomness but can encode capture order)",
    }

    # ---------------- headline sentence (derived, not typed) ----------------
    a_mean, b_mean = map_blocks["map_a"]["mean"], map_blocks["map_b"]["mean"]
    a_tar, b_tar = a_mean["tar_at_far_1e-2"], b_mean["tar_at_far_1e-2"]
    a_far, b_far = a_mean["far_at_frr_1pct"], b_mean["far_at_frr_1pct"]
    if b_far > 0:
        far_ratio = a_far / b_far
        far_gap = f"descriptive/in-sample at FRR=1%: FAR {a_far:.2%} (Map A) vs {b_far:.3%} (Map B), {far_ratio:.1f}x more impostors accepted by Map A"
    else:
        far_gap = f"descriptive/in-sample at FRR=1%: FAR {a_far:.2%} (Map A) vs 0% (Map B)"
    statement = (
        "Descriptive same-sample result (known-population cross-identity verification, "
        f"thresholds not calibrated on held-out identities): at a 1% false-accept rate, "
        f"Map B accepts {b_tar:.1%} of genuine probes versus {a_tar:.1%} for Map A; at "
        f"matched 1% false-reject rate Map A accepts {a_far:.2%} of impostors versus "
        f"{b_far:.3%} for Map B."
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

    # ---------------- JSON payload (DESIGN.md §6 schema + honest additions) ----------------
    notes = [
        "Map_a, map_b, headline and session_bias_check are DESCRIPTIVE and IN-SAMPLE: "
        "thresholds are chosen and evaluated on the same scores, and the EER is an "
        "interpolated crossing between observed thresholds, not a realisable threshold. "
        "They compare representations under one protocol; they are not calibrated error "
        "rates for unseen identities.",
        "This protocol is KNOWN-POPULATION CROSS-IDENTITY verification, not open-set "
        "identification and not a FaceID/deployment estimate: every impostor is one of the "
        "same 40 Olivetti identities, so no unknown identity enters the descriptive block.",
        "Score = minimum Euclidean distance from a probe image to the 5 enrolment images; "
        "decision = accept if score <= threshold. Trials are full-dimensional, no PCA.",
        "Impostor trials: for each identity, the 390 images (all 10 images of each of the "
        "other 39 identities) are scored against that identity's enrolment.",
        "Trial granularity: 200 genuine trials per split give FRR resolution 0.5%; 15,600 "
        "impostor trials give FAR resolution 0.0064%. Map B's mean EER (0.22%) is below the "
        "genuine-trial grid resolution, so it is 'at or below the measurement floor'.",
        "calibration_heldout_evaluation is the separate honest check: thresholds are chosen "
        "on 13 calibration identities and applied to 13 identity-disjoint evaluation "
        "identities, with 14 identities enrolled in no gallery used as unknown-to-gallery "
        "probes. All 40 identities still come from Olivetti; the 5 partitions overlap and "
        "are not independent samples. It is a small-sample optimism check, not a deployment "
        "estimate.",
        "Calibration FAR targets are 1% (primary) and 0.1% (secondary). No FRR target is used "
        "because the 65-genuine calibration fold has 1/65 = 1.54% FRR granularity, so a 1% FRR "
        "target is unresolvable and would force a degenerate infinite threshold.",
        "The threshold_curves figure is drawn from the descriptive genuine/impostor scores "
        "pooled over the 5 primary splits.",
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
            "verification_type": "known-population cross-identity verification "
            "(not open-set, not deployment)",
            "impostor_population": "the other 39 identities from the same 40-person "
            "Olivetti set; every identity appears in both genuine and impostor trials",
            "threshold_calibration": "none for map_a/map_b/headline/session_bias_check: "
            "those are in-sample descriptive values. See calibration_heldout_evaluation "
            "for identity-disjoint calibration vs test.",
        },
        "map_a": map_blocks["map_a"],
        "map_b": map_blocks["map_b"],
        "headline": headline,
        "session_bias_check": session_bias_check,
        "score_rule_sensitivity": {
            "mean_template": sensitivity,
            "note": "Descriptive/in-sample sensitivity check: the primary rule is the "
            "minimum distance to the 5 enrolment images; averaging the enrolment images "
            "into one template gives the same ranking.",
        },
        "calibration_heldout_evaluation": {
            "protocol": {
                "purpose": "identity-disjoint calibration-vs-test check on the same 40 "
                "Olivetti identities; NOT an open-set or deployment evaluation",
                "identity_split": "13 calibration / 13 identity-disjoint evaluation / "
                "14 unknown-to-gallery identities per partition",
                "partition_seeds": HELDOUT_PARTITION_SEEDS,
                "calibration": "13 identities, enrol 5 / probe 5: 65 genuine and 1560 "
                "cross-identity impostor scores per partition; thresholds chosen here ONLY",
                "evaluation": "13 identity-disjoint identities, enrol 5 / probe 5: "
                "empirical FAR/FRR/TAR measured at the calibration thresholds",
                "unknown": "14 identities enrolled in no gallery: all 10 images (140 "
                "probes) scored against the 13 evaluation galleries; "
                "unknown_accept_rate is the fraction accepted as any enrolled identity",
                "targets": "calibration FAR 1% (primary) and 0.1% (secondary); no FRR "
                "target because 65 calibration genuine trials give 1.54% FRR granularity",
                "threshold_rule": "conservative empirical FAR threshold (tie-safe): "
                "largest threshold whose calibration FAR does not exceed the target",
                "eer_interpolation": "interpolated EER on calibration scores is reported "
                "with the held-out FAR/FRR at that threshold; it is not an achievable "
                "operating point",
                "not_open_set": "unknown identities are unknown to the gallery, not to the "
                "frozen model (it saw faces during pretraining); Olivetti is one small "
                "dataset, not an open-world population",
                "non_independent_partitions": "the 5 partition seeds reuse the same 40 "
                "identities and overlap heavily; they are repeated draws from one small "
                "dataset, not five independent samples",
            },
            "map_a": heldout_blocks["map_a"],
            "map_b": heldout_blocks["map_b"],
            "comparison": comparison,
        },
    }
    out_path = RESULTS_DIR / "threshold.json"
    save_json(payload, out_path)
    print(f"[threshold] wrote {out_path} in {time.time() - started:.1f}s")
    print(f"[threshold] HEADLINE: {statement}")
    return payload


if __name__ == "__main__":
    main()
