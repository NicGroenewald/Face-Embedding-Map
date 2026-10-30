"""Hand-computed cases for ``fem.metrics``.

Why hand-computed
-----------------
These functions produce every verification number in the paper.  Testing them
against scikit-learn wrappers only proves the wrapper exists; small cases
whose answers can be worked out on paper prove the *conventions* (accept iff
distance <= threshold, FAR/FRR direction, conservative TAR threshold) are the
ones the analysis assumes.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem import metrics


# ---------------------------------------------------------------------------
# distances and the Euclidean <-> cosine relation
# ---------------------------------------------------------------------------
def test_pairwise_distances_hand_computed():
    A = np.array([[0.0, 0.0], [3.0, 4.0]])
    B = np.array([[1.0, 0.0], [0.0, 4.0]])
    expected = np.array([[1.0, 4.0], [np.sqrt(20.0), 3.0]])
    np.testing.assert_allclose(metrics.pairwise_distances(A, B), expected)


def test_pairwise_distances_square_and_validation():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    distances = metrics.pairwise_distances(X)
    assert distances.shape == (3, 3)
    np.testing.assert_allclose(np.diag(distances), 0.0)
    np.testing.assert_allclose(distances[0, 2], 2.0)
    with pytest.raises(ValueError):
        metrics.pairwise_distances(X, np.zeros((2, 3)))
    with pytest.raises(ValueError):
        metrics.pairwise_distances(np.zeros(3))


def test_cosine_from_euclidean_matches_dot_product_for_unit_vectors():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(7, 5))
    B = rng.normal(size=(9, 5))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)

    distances = metrics.pairwise_distances(A, B)
    np.testing.assert_allclose(metrics.cosine_from_euclidean(distances), A @ B.T, atol=1e-12)


def test_cosine_from_euclidean_known_values_and_scalar():
    assert metrics.cosine_from_euclidean(0.0) == 1.0
    assert metrics.cosine_from_euclidean(np.sqrt(2.0)) == pytest.approx(0.0, abs=1e-12)
    assert metrics.cosine_from_euclidean(2.0) == pytest.approx(-1.0)
    assert isinstance(metrics.cosine_from_euclidean(0.5), float)


# ---------------------------------------------------------------------------
# clustering / identification
# ---------------------------------------------------------------------------
def test_silhouette_hand_computed():
    # Two 1-D clusters with a=1 (within-cluster neighbour distance) and
    # b={9.5, 10.5} for the different points; s=(b-a)/max(a,b).
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    labels = np.array([0, 0, 1, 1])
    expected = np.mean(
        [
            (10.5 - 1.0) / 10.5,
            (9.5 - 1.0) / 9.5,
            (9.5 - 1.0) / 9.5,
            (10.5 - 1.0) / 10.5,
        ]
    )
    assert metrics.silhouette(X, labels) == pytest.approx(expected, abs=1e-9)


def test_knn_loo_perfect_and_single_error():
    X = np.array([[0.0], [0.1], [10.0], [10.1], [20.0], [10.2]])
    labels = np.array([0, 0, 1, 1, 2, 2])
    result = metrics.knn_loo(X, labels)

    assert result["accuracy"] == pytest.approx(5.0 / 6.0)
    assert result["errors"] == [5]
    assert result["predictions"][5] == 1  # nearest is label 1, not its own label 2
    assert result["top5"] == pytest.approx(1.0)


def test_knn_loo_validation():
    X = np.arange(6, dtype=float).reshape(3, 2)
    labels = np.array([0, 0, 1])
    with pytest.raises(ValueError):
        metrics.knn_loo(X, labels, k=3)  # k must be < n_samples
    with pytest.raises(ValueError):
        metrics.knn_loo(X, labels, k=0)


def test_pair_distances_counts_and_values():
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    labels = np.array([0, 0, 1, 1])
    genuine, impostor = metrics.pair_distances(X, labels)

    np.testing.assert_allclose(np.sort(genuine), [0.1, 0.1])
    np.testing.assert_allclose(np.sort(impostor), [9.9, 10.0, 10.0, 10.1])


def test_per_identity_accuracy():
    labels = np.array([0, 0, 0, 1, 1, 1])
    predictions = np.array([0, 0, 1, 1, 1, 1])
    np.testing.assert_allclose(metrics.per_identity_accuracy(labels, predictions), [2.0 / 3.0, 1.0])


# ---------------------------------------------------------------------------
# verification metrics
# ---------------------------------------------------------------------------
def test_roc_auc_hand_computed():
    assert metrics.roc_auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 1.0
    assert metrics.roc_auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 0.0
    # Identical distributions: every comparison is a tie -> 0.5.
    assert metrics.roc_auc(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(0.5)


def test_eer_separable_and_identical():
    equal_error, threshold = metrics.eer(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    assert equal_error == pytest.approx(0.0)
    assert threshold == pytest.approx(3.0)

    equal_error, threshold = metrics.eer(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert equal_error == pytest.approx(0.5)
    assert threshold == pytest.approx(0.0)


def test_eer_on_separable_gaussians_is_small_and_consistent():
    rng = np.random.default_rng(0)
    genuine = rng.normal(0.0, 1.0, 1000)
    impostor = rng.normal(4.0, 1.0, 1000)

    equal_error, threshold = metrics.eer(genuine, impostor)
    # Theoretical EER for N(0,1) vs N(4,1) is Phi(-2) ~= 0.0228; with 1000
    # samples per class the estimate should land near it.
    assert 0.01 < equal_error < 0.04
    assert 1.0 < threshold < 3.0

    far, frr = metrics.far_frr_at_threshold(genuine, impostor, threshold)
    assert abs(far - frr) < 0.02  # EER means FAR == FRR at that threshold


def test_tar_at_far_uses_conservative_empirical_threshold():
    genuine = np.full(10, 0.05)
    impostor = np.linspace(0.0, 0.99, 100)  # exactly the 0.00..0.99 grid

    # FAR target 0.1 -> accept the 10 closest impostors, largest is 0.09.
    assert metrics.tar_at_far(genuine, impostor, 0.1) == pytest.approx(1.0)
    # FAR target 0.001 -> 0.1 accepted impostors rounds down to none: FAR 0.
    assert metrics.tar_at_far(genuine, impostor, 0.001) == pytest.approx(0.0)
    assert metrics.tar_at_far(np.full(10, 0.5), impostor, 0.1) == 0.0
    assert metrics.tar_at_far(genuine, impostor, 1.0) == pytest.approx(1.0)

    with pytest.raises(ValueError):
        metrics.tar_at_far(genuine, impostor, 1.5)


def test_threshold_sweep_arrays_and_monotonicity():
    genuine = np.array([1.0, 2.0, 3.0])
    impostor = np.array([2.0, 3.0, 4.0])

    sweep = metrics.threshold_sweep(genuine, impostor, thresholds=np.array([0.0, 2.0, 3.0, 5.0]))
    np.testing.assert_allclose(sweep["thresholds"], [0.0, 2.0, 3.0, 5.0])
    # accept iff distance <= threshold
    np.testing.assert_allclose(sweep["far"], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    np.testing.assert_allclose(sweep["frr"], [1.0, 1.0 / 3.0, 0.0, 0.0])

    default = metrics.threshold_sweep(genuine, impostor)
    assert default["thresholds"].size == 256
    assert np.all(np.diff(default["far"]) >= 0.0)
    assert np.all(np.diff(default["frr"]) <= 0.0)
    assert default["far"][-1] == pytest.approx(1.0)
    assert default["frr"][-1] == pytest.approx(0.0)


def test_threshold_at_frr_and_operating_point():
    genuine = np.array([1.0, 2.0, 3.0, 4.0])
    impostor = np.array([5.0, 6.0])

    threshold = metrics.threshold_at_frr(genuine, impostor, 0.25)
    assert threshold == pytest.approx(3.0)
    far, frr = metrics.far_frr_at_threshold(genuine, impostor, threshold)
    assert frr == pytest.approx(0.25)
    assert far == pytest.approx(0.0)

    assert metrics.threshold_at_frr(genuine, impostor, 0.0) == np.inf
    assert metrics.threshold_at_frr(genuine, impostor, 1.0) < genuine.min()


def test_summary_hand_computed():
    genuine = np.array([1.0, 2.0, 3.0])
    impostor = np.array([5.0, 6.0, 7.0])
    result = metrics.summary(genuine, impostor)

    assert result["genuine"] == {
        "n": 3,
        "mean": pytest.approx(2.0),
        "std": pytest.approx(1.0),
        "min": pytest.approx(1.0),
        "max": pytest.approx(3.0),
    }
    assert result["impostor"]["mean"] == pytest.approx(6.0)
    assert result["d_prime"] == pytest.approx(4.0)  # 4 / sqrt((1+1)/2)
    assert result["auc"] == 1.0
    assert result["eer"] == pytest.approx(0.0)
    assert result["eer_threshold"] == pytest.approx(3.0)
    assert result["tar_at_far_1e-3"] == pytest.approx(1.0)
    assert result["tar_at_far_1e-2"] == pytest.approx(1.0)
