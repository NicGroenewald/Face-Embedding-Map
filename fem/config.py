"""Project-wide paths and constants.

Why a module instead of ad-hoc ``Path(__file__).parent`` in every script:
every runner, test and the plotting layer must agree on where data lives and
on the frozen experimental constants (seed, input resolution, embedding
dimension).  Deriving the paths from ``__file__`` rather than the current
working directory means ``python analysis/run_live.py`` behaves the same
whether it is launched from the repo root or from ``analysis/``.

Nothing in here imports numpy/torch, so it is safe to import from anywhere.
"""

from __future__ import annotations

from pathlib import Path

# ``config.py`` lives in ``<repo>/fem/config.py``: one parent up is the repo.
ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = ROOT / "data"
RESULTS_DIR: Path = ROOT / "results"
FIGURES_DIR: Path = ROOT / "figures"
SCANS_DIR: Path = ROOT / "saved-face-embeddings"
PAPER_DIR: Path = ROOT / "paper"
CACHE_DIR: Path = ROOT / ".cache"
DOCS_DIR: Path = ROOT / "docs"

#: Manifest that makes the raw ``.npy`` scan files self-describing.
SCANS_MANIFEST: Path = DATA_DIR / "scans.json"

#: Olivetti is small enough to be fetched on demand; when the cache exists
#: this path makes ``load_olivetti`` network-free.
OLIVETTI_HOME: Path = DATA_DIR / "sklearn"

#: One seed for every random choice in the project (PCA, splits, sampling).
SEED: int = 0

#: InceptionResnetV1 input resolution in pixels.
FACE_SIZE: int = 160
#: InceptionResnetV1 output width (already L2-normalised by the model).
EMBED_DIM: int = 512

#: Olivetti: 40 people x 10 photos, 64x64 greyscale.
IDENTITIES: int = 40
IMAGES_PER_IDENTITY: int = 10
OLIVETTI_IMAGE_SHAPE: tuple[int, int] = (64, 64)

#: Live study: 12 scans x 10 frames, single subject.
LIVE_SCANS: int = 12
FRAMES_PER_SCAN: int = 10


def ensure_dirs() -> None:
    """Create the output directories runners write into (idempotent).

    ``results/`` and ``figures/`` are git-tracked by name only, so a fresh
    checkout does not necessarily have them on disk.  Callers should not have
    to remember this.
    """
    for directory in (RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
