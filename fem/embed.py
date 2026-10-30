"""The frozen face-embedding model: InceptionResnetV1 pretrained on VGGFace2.

Why wrap it
-----------
The project rule is "no training, nothing fits".  Every Map B number comes
from this one checkpoint, so the model must be created in exactly one place:
``eval()`` mode, gradients off, output L2-normalised, and — on this Mac — the
embedder on MPS (the detector stays on CPU; see ``scripts/live_capture.py``).

``InceptionResnetV1.forward`` already ends with ``F.normalize(..., p=2)`` in
this version, so its 512-d outputs are unit vectors.  That is an invariant the
whole distance analysis relies on (Euclidean distance between unit vectors is
monotone in cosine similarity), which is why ``embed_tensor`` verifies it once
per batch instead of trusting a docstring.

The Olivetti embedding of 400 images is cached to ``results/embeddings_olivetti.npz``
so downstream steps (metrics, threshold analysis) never need the model again.
A cache is only a shortcut, never a source of truth: it carries provenance
(cache format, model tag, checkpoint SHA-256, dataset and preprocessing
fingerprints, torch version, sizes) and integrity metadata (row norms, finite
check).  If any of that disagrees with the current pipeline, or the stored rows
are not finite unit vectors, the cache is reported with a warning and rebuilt
rather than silently reused.
"""

from __future__ import annotations

import hashlib
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch.models.inception_resnet_v1 import get_torch_home

from .config import EMBED_DIM, FACE_SIZE, RESULTS_DIR, SEED
from .data import load_olivetti
from .preprocess import to_model_input

#: Tag stored in the cache so a stale cache from another checkpoint is caught.
MODEL_NAME = "InceptionResnetV1(vggface2)"

#: Basename of the checkpoint facenet-pytorch caches for ``pretrained="vggface2"``.
CHECKPOINT_FILENAME = "20180402-114759-vggface2.pt"

#: Bump when the *meaning* of the cached array changes (e.g. new column order).
CACHE_FORMAT_VERSION = 2

#: Human-readable preprocessing version; the behavioural fingerprint below is
#: what actually guards the cache, this string documents what it is guarding.
PREPROCESS_RECIPE = (
    "to_model_input:v1: greyscale->RGB; bilinear resize to FACE_SIZE "
    "antialias; x*255; (x-127.5)/128"
)

#: Provenance fields that must match exactly for a cache to be reusable.
_PROVENANCE_KEYS = (
    "cache_format",
    "model",
    "checkpoint_sha256",
    "dataset_sha256",
    "preprocess_sha256",
    "torch",
    "seed",
    "embed_dim",
    "face_size",
    "n_images",
)

#: Integrity metadata stored next to the matrix and re-checked on load.
_INTEGRITY_KEYS = ("row_norm_min", "row_norm_max", "all_finite")


def _default_device() -> str:
    """``"mps"`` on Apple silicon, ``"cpu"`` everywhere else (honest, not guessed)."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def checkpoint_path() -> Path:
    """Where facenet-pytorch looks for the VGGFace2 checkpoint."""
    return Path(get_torch_home()) / "checkpoints" / CHECKPOINT_FILENAME


@lru_cache(maxsize=1)
def _hash_file(path: str, size: int, mtime_ns: int) -> str:
    """SHA-256 of a file, cached by (path, size, mtime) so re-hashing is free."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_sha256() -> str:
    """Identity of the actual weights file (``"missing"`` if not cached)."""
    path = checkpoint_path()
    if not path.is_file():
        return "missing"
    stat = path.stat()
    return _hash_file(str(path), stat.st_size, stat.st_mtime_ns)


def dataset_fingerprint(images: np.ndarray, labels: np.ndarray) -> str:
    """SHA-256 over the evaluation set's shapes, dtypes, pixels and labels.

    Any change to the images (or their order/labels) changes the fingerprint,
    so embeddings computed on a different dataset cannot be served from cache.
    """
    images = np.ascontiguousarray(np.asarray(images))
    labels = np.ascontiguousarray(np.asarray(labels))
    digest = hashlib.sha256()
    digest.update(str(images.shape).encode())
    digest.update(str(images.dtype).encode())
    digest.update(images.tobytes())
    digest.update(str(labels.shape).encode())
    digest.update(str(labels.dtype).encode())
    digest.update(labels.tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def preprocess_fingerprint() -> str:
    """Behavioural fingerprint of ``to_model_input`` on a fixed probe.

    Hashing the actual output (not just a version string) means a change to
    resize mode, channel order, standardisation or the underlying resampling
    implementation invalidates cached embeddings even if nobody remembered to
    bump a version constant.
    """
    probe = np.linspace(0.0, 1.0, 2 * 64 * 64, dtype=np.float32).reshape(2, 64, 64)
    output = to_model_input(probe)
    digest = hashlib.sha256()
    digest.update(str(tuple(output.shape)).encode())
    digest.update(str(output.dtype).encode())
    digest.update(output.numpy().tobytes())
    digest.update(PREPROCESS_RECIPE.encode())
    return digest.hexdigest()


def cache_provenance(images: np.ndarray, labels: np.ndarray) -> dict:
    """Every field that must match for a cached embedding matrix to be reused."""
    return {
        "cache_format": CACHE_FORMAT_VERSION,
        "model": MODEL_NAME,
        "checkpoint_sha256": checkpoint_sha256(),
        "dataset_sha256": dataset_fingerprint(images, labels),
        "preprocess_sha256": preprocess_fingerprint(),
        "torch": str(torch.__version__),
        "seed": int(SEED),
        "embed_dim": int(EMBED_DIM),
        "face_size": int(FACE_SIZE),
        "n_images": int(np.asarray(images).shape[0]),
    }


def _as_python(value):
    """Normalise an ``.npz`` scalar into a plain Python value for comparison."""
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _cache_mismatches(cached: dict, expected: dict) -> list[str]:
    """Human-readable reasons why a cache cannot be reused (empty = compatible)."""
    reasons: list[str] = []
    for key in _PROVENANCE_KEYS:
        if key not in cached:
            reasons.append(f"{key}: missing from cache")
        elif cached[key] != expected[key]:
            reasons.append(f"{key}: cache={cached[key]!r}, expected={expected[key]!r}")
    return reasons


def _embedding_integrity(embeddings: np.ndarray) -> dict:
    """Row-norm/finiteness metadata written next to the cached matrix."""
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        "row_norm_min": float(norms.min()),
        "row_norm_max": float(norms.max()),
        "all_finite": bool(np.isfinite(embeddings).all()),
    }


def _validate_embeddings(embeddings: np.ndarray, n_expected: int | None = None) -> list[str]:
    """Problems with an embedding matrix as ``fem`` requires it (empty = valid)."""
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != EMBED_DIM:
        return [f"shape {array.shape}, expected (N, {EMBED_DIM})"]
    problems: list[str] = []
    if n_expected is not None and array.shape[0] != n_expected:
        problems.append(f"{array.shape[0]} rows, expected {n_expected}")
    if not np.isfinite(array).all():
        problems.append("contains non-finite values")
        return problems
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        problems.append(
            f"row norms outside 1±1e-4 (min={norms.min():.6f}, max={norms.max():.6f})"
        )
    return problems


def _read_embeddings_cache(
    path: Path, expected: dict
) -> tuple[np.ndarray | None, list[str]]:
    """Load a cache if it is compatible and internally consistent.

    Returns ``(embeddings, [])`` on success or ``(None, reasons)`` when the
    cache should be rebuilt.  This function never raises for a bad cache; it
    reports why so the caller can warn and recompute.
    """
    try:
        with np.load(path) as cached_file:
            keys = set(cached_file.files)
            if "embeddings" not in keys:
                return None, ["no 'embeddings' array in cache"]
            embeddings = np.asarray(cached_file["embeddings"], dtype=np.float32)
            cached = {key: _as_python(cached_file[key]) for key in keys if key != "embeddings"}
    except (OSError, ValueError, KeyError) as error:
        return None, [f"unreadable cache ({error})"]

    reasons = _cache_mismatches(cached, expected)
    missing_integrity = [key for key in _INTEGRITY_KEYS if key not in cached]
    if missing_integrity:
        reasons.append(f"integrity metadata missing: {missing_integrity}")
    if reasons:
        return None, reasons

    problems = _validate_embeddings(embeddings, expected["n_images"])
    if problems:
        return None, problems

    actual = _embedding_integrity(embeddings)
    for key in _INTEGRITY_KEYS:
        stored = cached[key]
        if key == "all_finite":
            matches = bool(stored) == actual[key]
        else:
            matches = abs(float(stored) - actual[key]) <= 1e-6
        if not matches:
            reasons.append(f"{key}: cache={stored!r}, actual={actual[key]!r}")
    if reasons:
        return None, reasons
    return embeddings, []


class FaceEmbedder:
    """Frozen InceptionResnetV1(vggface2); output rows are L2-normalised 512-d.

    Parameters
    ----------
    device:
        Torch device string.  Defaults to ``"mps"`` when available, else
        ``"cpu"``.  An explicitly requested unavailable device raises instead
        of silently falling back — a results JSON must not report ``mps``
        when the work happened on ``cpu``.
    """

    def __init__(self, device: str | None = None):
        self.device: str = str(device) if device is not None else _default_device()
        if self.device.startswith("mps") and not torch.backends.mps.is_available():
            raise ValueError("device='mps' requested but MPS is not available")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is not available")

        # pretrained='vggface2' resolves to the locally cached checkpoint; the
        # project forbids downloads during analysis runs.
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def embed_images(self, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Embed ``(N, 64, 64)`` float images in ``[0, 1]``.

        Returns ``(N, 512)`` float32 with unit-norm rows, in input order.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        model_input = to_model_input(images)
        if model_input.shape[-2:] != (FACE_SIZE, FACE_SIZE):
            raise ValueError(f"unexpected model input shape {tuple(model_input.shape)}")

        outputs: list[np.ndarray] = []
        n_images = model_input.shape[0]
        for start in range(0, n_images, batch_size):
            batch = model_input[start : start + batch_size]
            outputs.append(self.embed_tensor(batch))
        return np.concatenate(outputs, axis=0)

    def embed_tensor(self, x: torch.Tensor) -> np.ndarray:
        """Embed an already-prepared ``(N, 3, 160, 160)`` tensor.

        Kept separate from ``embed_images`` so a caller that already has a
        standardised tensor (or a batch of them) does not re-run preprocessing.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"embed_tensor expects a torch.Tensor, got {type(x).__name__}")
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected (N, 3, H, W) tensor, got shape {tuple(x.shape)}")

        with torch.no_grad():
            embeddings = self.model(x.to(self.device, dtype=torch.float32))

        result = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
        if result.ndim != 2 or result.shape[1] != EMBED_DIM:
            raise RuntimeError(f"model returned unexpected shape {result.shape}")
        norms = np.linalg.norm(result, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise RuntimeError(
                "InceptionResnetV1 output is not unit-norm "
                f"(min={norms.min():.6f}, max={norms.max():.6f}); "
                "the Euclidean<->cosine relation would be invalid"
            )
        return result

    def __repr__(self) -> str:
        return f"FaceEmbedder(device={self.device!r}, model={MODEL_NAME!r})"


def embed_olivetti(
    embedder: FaceEmbedder | None = None,
    cache: Path | None = None,
) -> np.ndarray:
    """Return ``(400, 512)`` Olivetti embeddings, using ``results/`` cache.

    The cache is keyed by the actual checkpoint hash, a dataset fingerprint, a
    behavioural preprocessing fingerprint, the torch version and the shapes in
    ``cache_provenance``.  A cache that does not match any of those — or whose
    stored rows fail the finite/unit-norm/integrity checks — is not silently
    trusted: it is reported with a ``UserWarning`` and rebuilt from the model.
    """
    cache_path = Path(cache) if cache is not None else RESULTS_DIR / "embeddings_olivetti.npz"

    # Needed for the dataset fingerprint; loading the cached Olivetti data is
    # fast and network-free, so validating provenance never costs a download.
    images, labels = load_olivetti()
    expected = cache_provenance(images, labels)

    if cache_path.exists():
        embeddings, reasons = _read_embeddings_cache(cache_path, expected)
        if embeddings is not None:
            return embeddings
        warnings.warn(
            f"ignoring incompatible embedding cache {cache_path} "
            f"({'; '.join(reasons)}); rebuilding",
            UserWarning,
            stacklevel=2,
        )

    embedder = embedder or FaceEmbedder()
    embeddings = np.asarray(embedder.embed_images(images), dtype=np.float32)
    problems = _validate_embeddings(embeddings, expected["n_images"])
    if problems:
        raise RuntimeError(
            "embedder returned an invalid embedding matrix: " + "; ".join(problems)
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        **expected,
        **_embedding_integrity(embeddings),
    )
    return embeddings


def load_cached_olivetti_embeddings(cache: Path | None = None) -> np.ndarray:
    """Load and validate the cached ``(400, 512)`` Olivetti embeddings.

    Read-only counterpart to ``embed_olivetti`` for analysis runners: it never
    runs the model and never rewrites the cache.  It is strict by design — a
    cache from another checkpoint, dataset, preprocessing recipe or torch
    version, or one with non-finite/non-unit rows, raises instead of being
    silently accepted (which would change every downstream Map B number).
    The error message tells the caller how to rebuild.

    Parameters
    ----------
    cache:
        Path to the ``.npz`` cache.  Defaults to
        ``results/embeddings_olivetti.npz``.

    Raises
    ------
    FileNotFoundError
        The cache file does not exist.
    ValueError
        The cache exists but fails provenance or integrity validation.
    """
    cache_path = Path(cache) if cache is not None else RESULTS_DIR / "embeddings_olivetti.npz"
    rebuild = f"run `make maps` (analysis/run_olivetti.py) to rebuild {cache_path}"

    if not cache_path.exists():
        raise FileNotFoundError(
            f"cached Olivetti embeddings not found at {cache_path}; {rebuild}"
        )

    images, labels = load_olivetti()
    expected = cache_provenance(images, labels)
    embeddings, reasons = _read_embeddings_cache(cache_path, expected)
    if embeddings is None:
        raise ValueError(
            f"cached Olivetti embeddings at {cache_path} are stale or invalid "
            f"({'; '.join(reasons)}); {rebuild}"
        )
    return embeddings
