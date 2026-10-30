"""The frozen face-embedding model: InceptionResnetV1 pretrained on VGGFace2.

Why wrap it
-----------
The project rule is "no training, nothing fits".  Every Map B number comes
from this one checkpoint, so the model must be created in exactly one place:
``eval()`` mode, gradients off, output L2-normalised, and — on this Mac — the
embedder on MPS (the detector stays on CPU; see ``live.py``).

``InceptionResnetV1.forward`` already ends with ``F.normalize(..., p=2)`` in
this version, so its 512-d outputs are unit vectors.  That is an invariant the
whole distance analysis relies on (Euclidean distance between unit vectors is
monotone in cosine similarity), which is why ``embed_tensor`` verifies it once
per batch instead of trusting a docstring.

The Olivetti embedding of 400 images is cached to ``results/embeddings_olivetti.npz``
so downstream steps (metrics, threshold analysis) never need the model again.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from .config import EMBED_DIM, FACE_SIZE, RESULTS_DIR, SEED
from .data import load_olivetti
from .preprocess import to_model_input

#: Tag stored in the cache so a stale cache from another checkpoint is caught.
MODEL_NAME = "InceptionResnetV1(vggface2)"


def _default_device() -> str:
    """``"mps"`` on Apple silicon, ``"cpu"`` everywhere else (honest, not guessed)."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


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

    The cache is keyed by model tag and seed in its ``.npz`` payload: the same
    deterministic model always produces the same vectors, so recomputing is
    wasted GPU time, but a cache written by a different checkpoint must not be
    silently reused.
    """
    cache_path = Path(cache) if cache is not None else RESULTS_DIR / "embeddings_olivetti.npz"

    if cache_path.exists():
        with np.load(cache_path) as cached:
            embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
            cached_model = str(cached["model"]) if "model" in cached else ""
            cached_seed = int(cached["seed"]) if "seed" in cached else SEED
        if cached_model and cached_model != MODEL_NAME:
            raise ValueError(
                f"cache {cache_path} was written by {cached_model!r}, "
                f"this run uses {MODEL_NAME!r}; delete the cache or pass a new path"
            )
        if cached_seed != SEED:
            raise ValueError(
                f"cache {cache_path} was written with seed {cached_seed}, "
                f"this run uses {SEED}; delete the cache or pass a new path"
            )
        if embeddings.shape != (400, EMBED_DIM):
            raise ValueError(
                f"cache {cache_path} holds shape {embeddings.shape}, expected (400, {EMBED_DIM})"
            )
        return embeddings

    images, _labels = load_olivetti()
    embedder = embedder or FaceEmbedder()
    embeddings = embedder.embed_images(images)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        model=MODEL_NAME,
        seed=np.int64(SEED),
        embed_dim=np.int64(EMBED_DIM),
        face_size=np.int64(FACE_SIZE),
    )
    return embeddings
