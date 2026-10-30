"""Metrics that turn a map into a number: clustering, identification,
verification (genuine/impostor), and threshold operating points.

Why one module
--------------
Every number in the paper comes from one of these functions.  The alternative
— each runner implementing its own ROC/EER curve — is how two tables in the
same paper end up computed slightly differently (plan trap #4: don't build the
pipeline twice).  Protocols (which pairs count as genuine) live in the
analysis runners; the *arithmetic* lives here.

Conventions
-----------
* Distances are Euclidean.  For unit vectors the cosine similarity is exactly
  ``1 - d**2 / 2``, available as ``cosine_from_euclidean``.
* A verifier accepts when ``distance <= threshold``.  Therefore:
    - FAR = fraction of impostor trials with distance <= threshold,
    - FRR = fraction of genuine trials with distance  > threshold,
    - ROC-AUC = P(genuine distance < impostor distance) with ties counted 1/2.
* ``eer`` interpolates the first FAR/FRR crossing on the exact threshold grid;
  ``tar_at_far`` uses the conservative empirical threshold (the largest
  threshold whose measured FAR does not exceed the target).
* EER, FAR and FRR are fractions in [0, 1], not percentages.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import pairwise_distances as _sklearn_pairwise_distances
from sklearn.metrics import roc_auc_score, silhouette_score


# ---------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------
def pairwise_distances(
    A: np.ndarray,
    B: np.ndarray | None = None,
    metric: str = "euclidean",
) -> np.ndarray:
    """Distance matrix between every row of ``A`` and every row of ``B``.

    With ``B=None`` this is the ``(N, N)`` matrix including the (zero)
    diagonal.  Delegates to scikit-learn so the metric exactly matches what
    ``silhouette`` and k-NN use internally.
    """
    a = np.asarray(A, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {a.shape}")
    b = None
    if B is not None:
        b = np.asarray(B, dtype=np.float64)
        if b.ndim != 2:
            raise ValueError(f"B must be 2-D, got shape {b.shape}")
        if b.shape[1] != a.shape[1]:
            raise ValueError(f"A has {a.shape[1]} features but B has {b.shape[1]}")
    return _sklearn_pairwise_distances(a, b, metric=metric)


def cosine_from_euclidean(d: np.ndarray | float) -> np.ndarray | float:
    """Cosine *similarity* from Euclidean distance, for unit vectors only.

    ``d**2 = ||a - b||**2 = 2 - 2 cos(a, b)`` so ``cos = 1 - d**2 / 2``.
    Returns 1 for identical vectors and 0 for orthogonal ones.  Scalar input
    returns a Python float so it can be JSON-serialised directly.
    """
    array = np.asarray(d, dtype=np.float64)
    similarity = 1.0 - 0.5 * array**2
    return float(similarity) if similarity.ndim == 0 else similarity


# ---------------------------------------------------------------------------
# clustering and identification
# ---------------------------------------------------------------------------
def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score on the FULL-dimensional representation.

    Must not be called on the 2-D PCA coordinates — the 2-D plot is for human
    eyes, and measuring it would be measuring the plotting settings.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    if X.ndim != 2 or X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must agree: X is (n_samples, n_features)")
    n_labels = np.unique(labels).size
    if not 2 <= n_labels < X.shape[0]:
        raise ValueError(f"silhouette needs 2..{X.shape[0] - 1} distinct labels, got {n_labels}")
    return float(silhouette_score(X, labels, metric="euclidean"))


def knn_loo(X: np.ndarray, labels: np.ndarray, k: int = 1) -> dict:
    """Leave-one-out k-NN identification (self always excluded).

    This answers "if you showed the map one new photo of a person, would its
    nearest neighbour be the same person?" — the identification counterpart to
    silhouette.

    Returns
    -------
    dict
        ``accuracy`` (float), ``top5`` (fraction of images whose true label is
        among the 5 nearest neighbours), ``predictions`` (list[int], one per
        sample) and ``errors`` (list[int], indices of misclassified samples).
        Lists rather than arrays so callers can serialise the result directly.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    n_samples = X.shape[0]
    if X.ndim != 2 or labels.shape[0] != n_samples:
        raise ValueError("X must be 2-D and have one row per label")
    if not 1 <= k < n_samples:
        raise ValueError(f"k={k} must be in 1..{n_samples - 1}")

    distances = pairwise_distances(X)
    np.fill_diagonal(distances, np.inf)  # leave-one-out: never predict yourself
    order = np.argsort(distances, axis=1, kind="stable")

    nearest_labels = labels[order[:, :k]]
    if k == 1:
        predictions = nearest_labels[:, 0].copy()
    else:
        predictions = np.empty(n_samples, dtype=labels.dtype)
        for row in range(n_samples):
            counts = Counter(nearest_labels[row].tolist())
            best_count = max(counts.values())
            tied = [label for label, count in counts.items() if count == best_count]
            # Ties are broken by the nearest neighbour, which is the first
            # label encountered in distance order.
            predictions[row] = next(
                label for label in nearest_labels[row] if label in tied
            )

    top5_keys = min(5, n_samples - 1)
    top5_neighbours = labels[order[:, :top5_keys]]
    correct = predictions == labels

    return {
        "accuracy": float(correct.mean()),
        "top5": float(np.mean([labels[i] in top5_neighbours[i] for i in range(n_samples)])),
        "predictions": [int(prediction) for prediction in predictions],
        "errors": [int(index) for index in np.flatnonzero(~correct)],
    }


def per_identity_accuracy(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Accuracy per identity, ordered by sorted identity label.

    Used for the "are the same identities hard for both maps?" analysis: the
    40 values are correlated across Map A and Map B.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    identities = np.unique(labels)
    return np.array(
        [float(np.mean(predictions[labels == identity] == identity)) for identity in identities],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# verification: genuine vs impostor trials
# ---------------------------------------------------------------------------
def pair_distances(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All unordered pairs split into (genuine, impostor) distance arrays.

    For the Olivetti protocol (400 images, 40 identities x 10) this yields
    exactly 1800 genuine and 78 000 impostor pairs.  Every pair is used once;
    no sampling and no protocol decisions here.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    n_samples = X.shape[0]
    if X.ndim != 2 or labels.shape[0] != n_samples:
        raise ValueError("X must be 2-D and have one row per label")

    distances = pairwise_distances(X)
    row_index, col_index = np.triu_indices(n_samples, k=1)
    pair_distance = distances[row_index, col_index]
    genuine_mask = labels[row_index] == labels[col_index]
    return pair_distance[genuine_mask].copy(), pair_distance[~genuine_mask].copy()


def _as_sorted(a: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(a, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError(f"{name} distances are empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} distances contain NaN or infinity")
    return np.sort(array)


def roc_auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Area under the ROC curve for a distance score (lower = more genuine)."""
    genuine_sorted = _as_sorted(genuine, "genuine")
    impostor_sorted = _as_sorted(impostor, "impostor")
    # Negate so genuine trials get the higher score; sklearn then handles ties.
    scores = np.concatenate([-genuine_sorted, -impostor_sorted])
    targets = np.concatenate(
        [np.ones(genuine_sorted.size, dtype=np.int8), np.zeros(impostor_sorted.size, dtype=np.int8)]
    )
    return float(roc_auc_score(targets, scores))


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate and the threshold at which FAR == FRR.

    The crossing is interpolated linearly between the two adjacent exact
    thresholds, which is why the result is stable without a random grid.
    Returns ``(eer_fraction, threshold_distance)``.
    """
    genuine_sorted = _as_sorted(genuine, "genuine")
    impostor_sorted = _as_sorted(impostor, "impostor")
    thresholds = np.unique(np.concatenate([genuine_sorted, impostor_sorted]))
    far, frr = _far_frr_at(thresholds, genuine_sorted, impostor_sorted)

    difference = far - frr  # negative while FAR < FRR, positive after crossing
    crossings = np.flatnonzero(difference >= 0)
    if crossings.size == 0:  # pragma: no cover - impossible with far(hi)=1, frr(hi)=0
        return 1.0, float(thresholds[-1])

    index = int(crossings[0])
    if index == 0:
        # Crossing is at or below the smallest observed distance: report the
        # average of the two rates there, which brackets the EER.
        return float(0.5 * (far[0] + frr[0])), float(thresholds[0])

    previous_difference = difference[index - 1]
    current_difference = difference[index]
    span = current_difference - previous_difference
    fraction = 0.0 if span == 0 else -previous_difference / span
    threshold = thresholds[index - 1] + fraction * (
        thresholds[index] - thresholds[index - 1]
    )
    equal_error = far[index - 1] + fraction * (far[index] - far[index - 1])
    return float(equal_error), float(threshold)


def _far_frr_at(
    thresholds: np.ndarray,
    genuine_sorted: np.ndarray,
    impostor_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """FAR/FRR at each threshold, assuming accept iff distance <= threshold."""
    far = np.searchsorted(impostor_sorted, thresholds, side="right") / impostor_sorted.size
    frr = 1.0 - np.searchsorted(genuine_sorted, thresholds, side="right") / genuine_sorted.size
    return far, frr


def tar_at_far(genuine: np.ndarray, impostor: np.ndarray, far: float) -> float:
    """True acceptance rate at a target false-acceptance rate.

    Uses the *conservative* empirical threshold: the largest threshold for
    which the observed impostor acceptance fraction does not exceed ``far``.
    This is why the reported TAR can be slightly pessimistic rather than
    optimistic, which is the safe direction for a security claim.
    """
    if not 0.0 <= far <= 1.0:
        raise ValueError(f"far must be a fraction in [0, 1], got {far}")
    genuine_sorted = _as_sorted(genuine, "genuine")
    impostor_sorted = _as_sorted(impostor, "impostor")

    if far <= 0.0:
        threshold = np.nextafter(impostor_sorted[0], -np.inf)
    elif far >= 1.0:
        threshold = np.inf
    else:
        accepted = int(np.floor(far * impostor_sorted.size))
        threshold = (
            impostor_sorted[accepted - 1]
            if accepted > 0
            else np.nextafter(impostor_sorted[0], -np.inf)
        )
    return float(np.mean(genuine_sorted <= threshold))


def far_frr_at_threshold(
    genuine: np.ndarray, impostor: np.ndarray, threshold: float
) -> tuple[float, float]:
    """(FAR, FRR) at one operating threshold; accept iff distance <= threshold."""
    genuine_sorted = _as_sorted(genuine, "genuine")
    impostor_sorted = _as_sorted(impostor, "impostor")
    far = float(np.searchsorted(impostor_sorted, threshold, side="right") / impostor_sorted.size)
    frr = float(
        1.0 - np.searchsorted(genuine_sorted, threshold, side="right") / genuine_sorted.size
    )
    return far, frr


def threshold_at_frr(genuine: np.ndarray, impostor: np.ndarray, frr: float) -> float:
    """Largest threshold whose observed FRR does not exceed the target.

    Used for the "convenience" operating point (e.g. reject at most 1 % of
    genuine probes) in the threshold analysis: pick the threshold here, then
    report the FAR it produces with ``far_frr_at_threshold``.
    """
    if not 0.0 <= frr <= 1.0:
        raise ValueError(f"frr must be a fraction in [0, 1], got {frr}")
    genuine_sorted = _as_sorted(genuine, "genuine")
    # The impostor sample is not needed to choose the threshold, but accepting
    # it keeps the function signature symmetric with the other helpers.
    _as_sorted(impostor, "impostor")

    if frr <= 0.0:
        return float(np.inf)  # accept every genuine probe
    if frr >= 1.0:
        return float(np.nextafter(genuine_sorted[0], -np.inf))  # reject all

    accepted = genuine_sorted.size - int(np.floor(frr * genuine_sorted.size))
    if accepted <= 0:
        return float(np.nextafter(genuine_sorted[0], -np.inf))
    if accepted >= genuine_sorted.size:
        return float(np.inf)
    return float(genuine_sorted[accepted - 1])


def threshold_sweep(
    genuine: np.ndarray,
    impostor: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    """FAR/FRR on a dense threshold grid, for the FAR/FRR-vs-threshold plot.

    Default grid: 256 evenly spaced points from 0 to the largest observed
    distance.  Passing explicit ``thresholds`` is useful when several curves
    must share an x-axis.
    """
    genuine_sorted = _as_sorted(genuine, "genuine")
    impostor_sorted = _as_sorted(impostor, "impostor")
    if thresholds is None:
        high = max(float(genuine_sorted[-1]), float(impostor_sorted[-1]))
        if high <= 0:
            high = 1.0
        thresholds = np.linspace(0.0, high, 256)
    thresholds = np.asarray(thresholds, dtype=np.float64).ravel()
    if thresholds.size == 0:
        raise ValueError("thresholds is empty")
    far, frr = _far_frr_at(thresholds, genuine_sorted, impostor_sorted)
    return {"thresholds": thresholds, "far": far, "frr": frr}


def summary(genuine: np.ndarray, impostor: np.ndarray) -> dict:
    """The full verification summary for one map's pair distances.

    Keys map one-to-one onto the ``results/olivetti.json`` schema (``pairs``
    and ``roc`` blocks), so the runner only has to splice this dict in.
    """
    genuine_sorted = _as_sorted(genuine, "genuine")
    impostor_sorted = _as_sorted(impostor, "impostor")

    def _stats(values: np.ndarray) -> dict:
        return {
            "n": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }

    var_genuine = genuine_sorted.var(ddof=1) if genuine_sorted.size > 1 else 0.0
    var_impostor = impostor_sorted.var(ddof=1) if impostor_sorted.size > 1 else 0.0
    pooled_sd = float(np.sqrt(0.5 * (var_genuine + var_impostor)))
    if pooled_sd > 0:
        d_prime = float(abs(impostor_sorted.mean() - genuine_sorted.mean()) / pooled_sd)
    else:
        # Degenerate zero-variance case: infinite separation if means differ.
        d_prime = float("inf") if impostor_sorted.mean() != genuine_sorted.mean() else 0.0

    equal_error_rate, equal_error_threshold = eer(genuine_sorted, impostor_sorted)
    return {
        "genuine": _stats(genuine_sorted),
        "impostor": _stats(impostor_sorted),
        "d_prime": d_prime,
        "auc": roc_auc(genuine_sorted, impostor_sorted),
        "eer": equal_error_rate,
        "eer_threshold": equal_error_threshold,
        "tar_at_far_1e-3": tar_at_far(genuine_sorted, impostor_sorted, 1e-3),
        "tar_at_far_1e-2": tar_at_far(genuine_sorted, impostor_sorted, 1e-2),
    }
