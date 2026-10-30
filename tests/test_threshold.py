"""Tests for the identity-disjoint verification analysis in ``analysis/run_threshold.py``.

Why these tests exist
---------------------
Task-11's point is scientific honesty: the descriptive in-sample block must not
be sold as calibrated, and the additive ``calibration_heldout_evaluation`` block
must genuinely choose thresholds on calibration identities and measure error
rates on identities the thresholds never saw.  That property is easy to lose in
a refactor (e.g. by accidentally reusing evaluation scores when choosing a
threshold), and it cannot be checked by reading the JSON, so it is pinned here
with small, hand-computable cases.

No model, no network, no webcam, no results/ writes: everything here operates on
synthetic distance matrices or on pure helper functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis import run_threshold as threshold


# ---------------------------------------------------------------------------
# identity partitioning
# ---------------------------------------------------------------------------
def test_partition_identities_is_deterministic_disjoint_and_complete():
    cal, evaluation, unknown = threshold._partition_identities(0)
    assert (cal.size, evaluation.size, unknown.size) == (13, 13, 14)

    for seed in threshold.HELDOUT_PARTITION_SEEDS:
        cal, evaluation, unknown = threshold._partition_identities(seed)
        # disjoint folds, union is every identity exactly once
        assert not set(cal) & set(evaluation)
        assert not set(cal) & set(unknown)
        assert not set(evaluation) & set(unknown)
        union = np.sort(np.concatenate([cal, evaluation, unknown]))
        np.testing.assert_array_equal(union, np.arange(threshold.N_IDENTITIES))

    # same seed -> same partition; different seeds -> genuinely different draws
    first = threshold._partition_identities(0)
    for array, again in zip(first, threshold._partition_identities(0)):
        np.testing.assert_array_equal(array, again)
    draws = [threshold._partition_identities(seed)[0] for seed in threshold.HELDOUT_PARTITION_SEEDS]
    assert any(not np.array_equal(draws[0], other) for other in draws[1:])


def test_partition_identities_small_and_validation():
    cal, evaluation, unknown = threshold._partition_identities(0, n_identities=6)
    assert (cal.size, evaluation.size, unknown.size) == (2, 2, 2)
    cal, evaluation, unknown = threshold._partition_identities(0, n_identities=7)
    assert (cal.size, evaluation.size, unknown.size) == (2, 2, 3)
    with pytest.raises(ValueError):
        threshold._partition_identities(0, n_identities=5)


def test_enrolment_split_is_disjoint_covering_and_seeded():
    identities = np.array([3, 7])
    enrol, probe = threshold._enrolment_split(
        identities, seed=0, images_per_identity=10, enrol=5
    )
    for identity in identities:
        assert enrol[identity].size == 5
        assert probe[identity].size == 5
        assert not set(enrol[identity]) & set(probe[identity])
        np.testing.assert_array_equal(enrol[identity], np.sort(enrol[identity]))
        np.testing.assert_array_equal(probe[identity], np.sort(probe[identity]))
    covered = np.sort(
        np.concatenate([np.concatenate([enrol[i], probe[i]]) for i in identities])
    )
    expected = np.sort(np.concatenate([np.arange(30, 40), np.arange(70, 80)]))
    np.testing.assert_array_equal(covered, expected)

    enrol_again, probe_again = threshold._enrolment_split(
        identities, seed=0, images_per_identity=10, enrol=5
    )
    for identity in identities:
        np.testing.assert_array_equal(enrol[identity], enrol_again[identity])
        np.testing.assert_array_equal(probe[identity], probe_again[identity])

    enrol_other, _ = threshold._enrolment_split(
        identities, seed=1, images_per_identity=10, enrol=5
    )
    assert any(not np.array_equal(enrol[i], enrol_other[i]) for i in identities)

    with pytest.raises(ValueError):
        threshold._enrolment_split(identities, seed=0, images_per_identity=10, enrol=10)


# ---------------------------------------------------------------------------
# conservative threshold rules (hand-computed)
# ---------------------------------------------------------------------------
def test_threshold_at_far_hand_computed():
    impostor = np.array([1.0, 2.0, 3.0, 4.0])
    # floor(target * n): 1, 2, 4 accepted impostors respectively
    assert threshold._threshold_at_far(impostor, 0.25) == 1.0
    assert threshold._threshold_at_far(impostor, 0.5) == 2.0
    assert threshold._threshold_at_far(impostor, 1.0) == np.inf
    assert threshold._threshold_at_far(impostor, 0.0) < 1.0

    # The promise of the rule: observed FAR never exceeds the requested target.
    for target in (0.0, 0.1, 0.25, 0.5, 0.99):
        chosen = threshold._threshold_at_far(impostor, target)
        assert np.mean(impostor <= chosen) <= target + 1e-12


def test_threshold_at_far_handles_ties_conservatively():
    # Both distinct score values already accept 2/4 = 50% of impostors, so a
    # 25% target has no feasible score value: the threshold must drop below all
    # scores rather than accept a tie that overshoots the target.
    impostor = np.array([1.0, 1.0, 2.0, 2.0])
    chosen = threshold._threshold_at_far(impostor, 0.25)
    assert chosen < 1.0
    assert np.mean(impostor <= chosen) == 0.0
    # A 50% target is satisfiable exactly at 1.0 (accepted count 2).
    assert threshold._threshold_at_far(impostor, 0.5) == 1.0

    with pytest.raises(ValueError):
        threshold._threshold_at_far(impostor, 1.5)
    with pytest.raises(ValueError):
        threshold._threshold_at_far(np.array([]), 0.1)


def test_heldout_targets_are_far_only():
    # Lead-approved design: calibration FAR 1% and 0.1%; no FRR target, because
    # the 65-genuine calibration fold has 1.54% FRR granularity.
    assert threshold.HELDOUT_TARGETS == (("far", 1e-2), ("far", 1e-3))
    assert all(rule == "far" for rule, _ in threshold.HELDOUT_TARGETS)


# ---------------------------------------------------------------------------
# score semantics (hand-computed)
# ---------------------------------------------------------------------------
def test_pair_scores_hand_computed():
    distances = np.array(
        [
            [0.0, 1.0, 5.0, 6.0],
            [1.0, 0.0, 7.0, 8.0],
            [5.0, 7.0, 0.0, 2.0],
            [6.0, 8.0, 2.0, 0.0],
        ]
    )
    enrol = {0: np.array([0]), 1: np.array([2])}
    probe = {0: np.array([1]), 1: np.array([3])}
    genuine, impostor = threshold._pair_scores(
        distances, np.array([0, 1]), enrol, probe
    )
    # genuine: d(probe 1 -> enrol 0) = 1, d(probe 3 -> enrol 2) = 2
    np.testing.assert_allclose(np.sort(genuine), [1.0, 2.0])
    # identity 0: others' images {2, 3} vs enrol 0 -> [5, 6]
    # identity 1: others' images {0, 1} vs enrol 2 -> [5, 7]
    np.testing.assert_allclose(np.sort(impostor), [5.0, 5.0, 6.0, 7.0])


def test_unknown_scores_use_nearest_gallery_identity():
    distances = np.array(
        [
            [0.0, 9.0, 0.1, 0.5],
            [9.0, 0.0, 0.2, 0.4],
            [0.1, 0.2, 0.0, 8.0],
            [0.5, 0.4, 8.0, 0.0],
        ]
    )
    enrol = {0: np.array([0]), 1: np.array([1])}
    scores = threshold._unknown_scores(
        distances, np.array([0, 1]), enrol, np.array([1]), images_per_identity=2
    )
    # unknown identity 1 images are indices 2 and 3; for each, take the closest
    # enrolled gallery identity: min over galleries of the min over enrol images
    np.testing.assert_allclose(scores, [0.1, 0.4])


# ---------------------------------------------------------------------------
# held-out evaluation wiring
# ---------------------------------------------------------------------------
def _toy_distances() -> np.ndarray:
    """6 identities x 2 images: distances from 1-D points (a valid metric matrix)."""
    points = np.array([0.0, 0.1, 1.0, 1.1, 2.0, 2.2, 3.0, 3.1, 5.0, 5.2, 6.0, 6.1])
    return np.abs(points[:, None] - points[None, :])


def test_heldout_evaluation_counts_and_rate_definitions():
    distances = _toy_distances()
    public, internal = threshold._heldout_evaluation(
        distances, seed=0, n_identities=6, images_per_identity=2, enrol=1
    )
    assert public["n_calibration_identities"] == 2
    assert public["n_evaluation_identities"] == 2
    assert public["n_unknown_identities"] == 2
    # 2 calibration identities x 1 probe each
    assert public["n_calibration_genuine"] == 2
    # each calibration identity is scored against the other identity's 2 images
    assert public["n_calibration_impostor"] == 4
    assert public["n_evaluation_genuine"] == 2
    assert public["n_evaluation_impostor"] == 4
    # 2 unknown identities x 2 images, each scored against any evaluation gallery
    assert public["n_unknown_probes"] == 4

    # folds are disjoint and complete
    folds = [
        set(internal["calibration_identities"]),
        set(internal["evaluation_identities"]),
        set(internal["unknown_identities"]),
    ]
    assert folds[0].isdisjoint(folds[1]) and folds[0].isdisjoint(folds[2])
    assert folds[1].isdisjoint(folds[2])
    assert len(folds[0] | folds[1] | folds[2]) == 6

    for point in public["operating_points"]:
        chosen = point["threshold"]
        assert point["rule"] == "calibration_far"
        assert point["calibration_far"] <= point["target"] + 1e-12
        assert np.isclose(
            point["calibration_far"], np.mean(internal["impostor_calibration"] <= chosen)
        )
        assert np.isclose(
            point["evaluation_far"], np.mean(internal["impostor_evaluation"] <= chosen)
        )
        assert np.isclose(
            point["evaluation_frr"], np.mean(internal["genuine_evaluation"] > chosen)
        )
        assert np.isclose(
            point["evaluation_tar"], np.mean(internal["genuine_evaluation"] <= chosen)
        )
        assert np.isclose(
            point["unknown_accept_rate"], np.mean(internal["unknown_scores"] <= chosen)
        )
        # the threshold is exactly what the calibration scores alone justify
        assert np.isclose(
            chosen, threshold._threshold_at_far(internal["impostor_calibration"], point["target"])
        )


def test_heldout_thresholds_ignore_evaluation_scores():
    """Changing held-out identities' distances must not move the chosen thresholds."""
    distances = _toy_distances()
    seed = 0
    _, evaluation_ids, _ = threshold._partition_identities(seed, n_identities=6)
    evaluation_images = np.concatenate(
        [np.arange(int(identity) * 2, int(identity) * 2 + 2) for identity in evaluation_ids]
    )
    perturbed = distances.copy()
    perturbed[np.ix_(evaluation_images, np.arange(distances.shape[0]))] *= 3.0
    perturbed[np.ix_(np.arange(distances.shape[0]), evaluation_images)] *= 3.0

    public, _ = threshold._heldout_evaluation(
        distances, seed, n_identities=6, images_per_identity=2, enrol=1
    )
    public_perturbed, _ = threshold._heldout_evaluation(
        perturbed, seed, n_identities=6, images_per_identity=2, enrol=1
    )
    for original, changed in zip(
        public["operating_points"], public_perturbed["operating_points"]
    ):
        assert original["threshold"] == changed["threshold"]


def test_heldout_eer_interpolation_check_reports_calibration_eer_and_heldout_rates():
    public, internal = threshold._heldout_evaluation(
        _toy_distances(), seed=0, n_identities=6, images_per_identity=2, enrol=1
    )
    from fem import metrics

    expected_eer, expected_threshold = metrics.eer(
        internal["genuine_calibration"], internal["impostor_calibration"]
    )
    check = public["eer_interpolation_check"]
    assert np.isclose(check["calibration_eer_interpolated"], expected_eer)
    assert np.isclose(check["calibration_eer_threshold"], expected_threshold)
    # The check is honest about the interpolation: it reports what actually
    # happens on held-out data at the interpolated threshold.
    assert np.isclose(
        check["evaluation_far_at_calibration_eer_threshold"],
        np.mean(internal["impostor_evaluation"] <= expected_threshold),
    )
    assert np.isclose(
        check["evaluation_frr_at_calibration_eer_threshold"],
        np.mean(internal["genuine_evaluation"] > expected_threshold),
    )
    assert "not a realisable operating point" in check["note"]


def test_aggregate_heldout_mean_and_std():
    def partition(threshold_value: float, tar: float) -> dict:
        return {
            "operating_points": [
                {
                    "rule": "calibration_far",
                    "target": 0.01,
                    "threshold": threshold_value,
                    "calibration_far": 0.0,
                    "calibration_frr": 0.2,
                    "evaluation_far": 0.0,
                    "evaluation_frr": 0.4,
                    "evaluation_tar": tar,
                    "unknown_accept_rate": 0.0,
                }
            ],
            "eer_interpolation_check": {
                "calibration_eer_interpolated": 0.5,
                "calibration_eer_threshold": 1.0,
                "evaluation_far_at_calibration_eer_threshold": 0.1,
                "evaluation_frr_at_calibration_eer_threshold": 0.2,
            },
        }

    per_partition = [partition(1.0, 0.6), partition(3.0, 0.8)]
    mean, std = threshold._aggregate_heldout(per_partition)
    point_mean = mean["operating_points"][0]
    point_std = std["operating_points"][0]
    assert point_mean["rule"] == "calibration_far"
    assert point_mean["target"] == 0.01
    assert point_mean["threshold"] == pytest.approx(2.0)
    assert point_mean["evaluation_tar"] == pytest.approx(0.7)
    assert point_std["threshold"] == pytest.approx(np.std([1.0, 3.0], ddof=1))
    assert std["eer_interpolation_check"]["calibration_eer_interpolated"] == pytest.approx(0.0)
