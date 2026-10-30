"""Tests for ``fem.maps`` using synthetic data with a known structure.

Why synthetic: PCA's job in this project is to recover a low-dimensional
subspace from high-dimensional data.  A generated rank-2 matrix has a
mathematically exact answer — the projection must capture 100 % of the
variance and reproduce the data — which no real dataset can assert as cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem import maps


def rank_two_data(seed: int = 0, n_samples: int = 200, n_features: int = 10) -> np.ndarray:
    """A matrix that is exactly rank 2 in a 10-dimensional ambient space."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(n_samples, 2)) * np.array([3.0, 1.0])
    loadings = rng.normal(size=(2, n_features))
    return scores @ loadings


def test_pca_map_recovers_low_rank_subspace():
    X = rank_two_data()
    mapped = maps.pca_map(X, n_components=2)

    assert isinstance(mapped, maps.PcaMap)
    assert mapped.coords.shape == (200, 2)
    assert mapped.evr.shape == (2,)
    assert mapped.model.components_.shape == (2, 10)
    np.testing.assert_allclose(mapped.components, mapped.model.components_)

    # Rank-2 data is reproduced exactly by two components...
    assert mapped.evr.sum() == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(mapped.model.inverse_transform(mapped.coords), X, atol=1e-7)

    # ...so the 2-D distances are a monotone (in fact exact) view of the
    # full-dimensional distances.
    assert maps.rank_correlation_2d(mapped.coords, X) > 0.999


def test_explained_variance_is_ordered_and_matches_pca_map():
    X = rank_two_data(seed=1)
    evr = maps.explained_variance(X, n_components=5)
    assert evr.shape == (5,)
    assert np.all(np.diff(evr) <= 1e-12)
    assert evr.sum() == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(evr, maps.pca_map(X, n_components=5).evr)


def test_pca_map_validates_input():
    X = rank_two_data()
    with pytest.raises(ValueError):
        maps.pca_map(X, n_components=0)
    with pytest.raises(ValueError):
        maps.pca_map(X, n_components=11)
    with pytest.raises(ValueError):
        maps.pca_map(X.ravel())
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        maps.pca_map(bad)


def test_variance_milestones_hand_computed():
    evr = np.array([0.5, 0.3, 0.15, 0.05])
    assert maps.variance_milestones(evr) == {25: 1, 50: 1, 75: 2, 90: 3, 95: 3}
    assert maps.variance_milestones(evr, milestones=(100,)) == {100: 4}
    # Not reached within the supplied components -> explicit None.
    assert maps.variance_milestones(np.array([0.1, 0.1]), milestones=(95,)) == {95: None}
    with pytest.raises(ValueError):
        maps.variance_milestones(evr, milestones=(0,))


def test_eigenfaces_reshape_and_validation():
    rng = np.random.default_rng(0)
    components = rng.normal(size=(16, 4096))
    faces = maps.eigenfaces(components, n=4)
    assert faces.shape == (4, 64, 64)
    np.testing.assert_array_equal(faces[0], components[0].reshape(64, 64))

    with pytest.raises(ValueError):
        maps.eigenfaces(components, n=17)
    with pytest.raises(ValueError):
        maps.eigenfaces(rng.normal(size=(16, 100)), n=4)


def test_rank_correlation_2d_identical_and_independent():
    rng = np.random.default_rng(0)

    # Projection is the identity: distances are equal -> correlation 1.
    coords = rng.normal(size=(10, 2))
    assert maps.rank_correlation_2d(coords, coords) == pytest.approx(1.0)

    # Independent random coordinates vs random full vectors: no relationship.
    coords = rng.normal(size=(400, 2))
    full = rng.normal(size=(400, 5))
    correlation = maps.rank_correlation_2d(coords, full, n_pairs=20000, seed=0)
    assert abs(correlation) < 0.1


def test_rank_correlation_2d_uses_all_pairs_when_small_and_validates():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(5, 2))
    full = rng.normal(size=(5, 3))
    all_pairs = maps.rank_correlation_2d(coords, full, n_pairs=10)
    extra_pairs = maps.rank_correlation_2d(coords, full, n_pairs=20000)
    assert all_pairs == pytest.approx(extra_pairs)

    with pytest.raises(ValueError):
        maps.rank_correlation_2d(coords, rng.normal(size=(4, 3)))
