"""The two "maps": PCA projections of a full-dimensional representation.

Why PCA and not t-SNE/UMAP (the project's central trap)
-------------------------------------------------------
A *map* in this project is a reusable transform: fit on a representation, keep
the components, project new points with the same recipe.  PCA has that
property; t-SNE is a plotter whose "components" mean nothing outside the run.
The 2-D picture is for human eyes only — every metric is computed on the
full-dimensional vectors, because measuring a 2-D projection measures the
plotting settings (plan trap #3).

Why no StandardScaler
---------------------
All 4096 raw-pixel columns share one unit (brightness 0..1); per-column
scaling only amplifies near-constant background pixels into noise.  This was
tested and rejected in ``docs/findings/map-a-raw-pixels.md`` §4, so ``pca_map``
fits PCA on centred raw data and nothing else.

The same function is applied to Map A (4096-d pixels) and Map B (512-d
embeddings) with identical settings, so any difference between the maps comes
from the representation, not the recipe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from .config import SEED


@dataclass
class PcaMap:
    """A fitted PCA model plus the 2-D coordinates it produced.

    ``model`` is kept (not just ``coords``) so the same transform can project
    held-out points and so the components can be reshaped into eigenfaces.
    """

    model: PCA
    coords: np.ndarray  # (N, n_components)
    evr: np.ndarray  # explained_variance_ratio_, length n_components

    @property
    def components(self) -> np.ndarray:
        """The principal axes, ``(n_components, n_features)``.

        Convenience alias for ``model.components_`` so ``eigenfaces`` and
        component-row plots can be written without reaching into sklearn.
        """
        return self.model.components_


def _fit_pca(X: np.ndarray, n_components: int, seed: int) -> PCA:
    """Fit the one PCA configuration used everywhere in the project.

    ``svd_solver="randomized"`` is mandated by the build contract: it is fast
    on 4096-d pixels and 512-d embeddings, and the seed makes the projection
    reproducible.  Input validation lives here so every public function that
    fits PCA gets the same guarantees.
    """
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"PCA expects a 2-D array (n_samples, n_features), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("PCA input contains NaN or infinity")
    max_components = min(array.shape)
    if not 1 <= n_components <= max_components:
        raise ValueError(
            f"n_components={n_components} must be in 1..{max_components} for input {array.shape}"
        )
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    return pca.fit(array)


def pca_map(X: np.ndarray, n_components: int = 2, seed: int = SEED) -> PcaMap:
    """Fit PCA and return the projection (coords) and explained variance.

    Centring is sklearn's default and is the only preprocessing applied — see
    the module docstring for why scaling is deliberately omitted.
    """
    pca = _fit_pca(X, n_components=n_components, seed=seed)
    array = np.asarray(X, dtype=np.float64)
    return PcaMap(
        model=pca,
        coords=pca.transform(array),
        evr=np.asarray(pca.explained_variance_ratio_, dtype=np.float64).copy(),
    )


def explained_variance(X: np.ndarray, n_components: int = 100, seed: int = SEED) -> np.ndarray:
    """Explained-variance ratio for the first ``n_components`` components.

    Note the sum is *not* 1.0 unless ``n_components`` covers the whole rank —
    that gap is exactly the "what am I throwing away by squashing to 2-D"
    question, so the returned ratios must not be renormalised.
    """
    pca = _fit_pca(X, n_components=n_components, seed=seed)
    return np.asarray(pca.explained_variance_ratio_, dtype=np.float64).copy()


def variance_milestones(
    evr: np.ndarray,
    milestones: tuple[int, ...] = (25, 50, 75, 90, 95),
) -> dict[int, int | None]:
    """How many components are needed to reach each cumulative-variance target.

    Returns ``{milestone_percent: n_components}``; a milestone that is not
    reached within the supplied components maps to ``None``.  That ``None`` is
    information, not a bug: it says the scree plot was truncated before the
    target and the paper must say so (e.g. 95 % is not reached in Map A's
    first 100 components).
    """
    ratios = np.asarray(evr, dtype=np.float64).ravel()
    if ratios.size == 0:
        raise ValueError("evr is empty")
    cumulative = np.cumsum(ratios)
    result: dict[int, int | None] = {}
    for milestone in milestones:
        if not 0 < milestone <= 100:
            raise ValueError(f"milestone {milestone} is not a percentage in (0, 100]")
        # Tiny tolerance so a cumulative value that is mathematically equal to
        # the target (e.g. 0.5 + 0.3 + 0.15 = 0.95 in floats) is not missed.
        index = int(np.searchsorted(cumulative + 1e-12, milestone / 100.0, side="left"))
        result[int(milestone)] = int(index + 1) if index < cumulative.size else None
    return result


def eigenfaces(
    components: np.ndarray,
    n: int = 16,
    image_shape: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Reshape the first ``n`` principal components back into images.

    The components live in the same space as the flattened images, so a
    64x64 image with 4096 pixels maps one component to one 64x64 image.  Raw
    values are returned (not scaled); the plotting layer chooses the display
    range, because how to render a signed component is a presentation choice.
    """
    array = np.asarray(components, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"components must be 2-D, got shape {array.shape}")
    height, width = image_shape
    if array.shape[1] != height * width:
        raise ValueError(
            f"components have {array.shape[1]} values but image_shape={image_shape} "
            f"needs {height * width}"
        )
    if not 1 <= n <= array.shape[0]:
        raise ValueError(f"n={n} must be in 1..{array.shape[0]}")
    return array[:n].reshape(n, height, width).copy()


def rank_correlation_2d(
    coords: np.ndarray,
    X_full: np.ndarray,
    n_pairs: int = 20000,
    seed: int = SEED,
) -> float:
    """Spearman correlation of 2-D vs full-dimensional pairwise distances.

    This is the honest answer to "how much of the true geometry does the
    scatter plot preserve?".  Pearson correlation is not appropriate because
    the relationship between a 2-D projection distance and the full distance
    is monotone, not linear.

    For efficiency only ``n_pairs`` random pairs are compared (all pairs when
    the dataset is small enough).  The sample is seeded, so the number is
    reproducible.
    """
    from scipy.stats import spearmanr

    coords_array = np.asarray(coords, dtype=np.float64)
    full_array = np.asarray(X_full, dtype=np.float64)
    if coords_array.ndim != 2 or full_array.ndim != 2:
        raise ValueError("coords and X_full must both be 2-D")
    n_samples = coords_array.shape[0]
    if full_array.shape[0] != n_samples:
        raise ValueError(
            f"coords has {n_samples} rows but X_full has {full_array.shape[0]}"
        )
    if n_samples < 3:
        raise ValueError("need at least 3 samples to correlate pairwise distances")

    row_index, col_index = np.triu_indices(n_samples, k=1)
    total_pairs = row_index.size
    rng = np.random.default_rng(seed)
    if n_pairs >= total_pairs:
        chosen = np.arange(total_pairs)
    else:
        chosen = rng.choice(total_pairs, size=n_pairs, replace=False)

    rows, cols = row_index[chosen], col_index[chosen]
    distance_2d = np.linalg.norm(coords_array[rows] - coords_array[cols], axis=1)
    distance_full = np.linalg.norm(full_array[rows] - full_array[cols], axis=1)

    statistic = spearmanr(distance_2d, distance_full).statistic
    if not np.isfinite(statistic):
        raise ValueError(
            "Spearman correlation is undefined (one distance set is constant); "
            "check that the projection is not collapsed to a point"
        )
    return float(statistic)
