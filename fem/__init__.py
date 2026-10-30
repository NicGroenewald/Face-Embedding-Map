"""fem — the Face Embedding Map library.

One implementation of the project's pipeline:

    load  ->  embed / raw pixels  ->  PCA map  ->  metrics  ->  figures

Why a package
-------------
The project's own rule (``docs/PLAN.md`` trap #4) is "don't build the
pipeline twice".  Map A (raw pixels) and Map B (learned embeddings) differ
only in the representation; loading, PCA, metrics and plotting are shared.
``analysis/`` runners orchestrate these modules; they do not re-implement
them.

Module map
----------
``config``      paths + frozen constants (seed, sizes)
``data``        Olivetti and live-scan loaders (manifest-driven)
``preprocess``  64x64 Olivetti -> 160x160 standardised model input
``embed``       frozen InceptionResnetV1(vggface2), cached Olivetti embeddings
``maps``        PCA projection, eigenfaces, 2-D-vs-full geometry check
``metrics``     silhouette, k-NN LOO, genuine/impostor ROC/EER/thresholds
``plotting``    every figure saver (PNG 200 dpi + vector PDF)

Import policy
-------------
``import fem`` stays cheap: numpy/scikit-learn modules (``data``, ``maps``,
``metrics``) are eager, while ``preprocess``, ``embed`` and ``plotting`` are
loaded on first access.  That keeps ``import fem`` free of torch/matplotlib
side effects, which matters for tests and for the manifest-only script
``scripts/compare_scans.py``.
"""

from __future__ import annotations

import importlib

from . import data, maps, metrics
from .config import (
    DATA_DIR,
    EMBED_DIM,
    FACE_SIZE,
    FIGURES_DIR,
    FRAMES_PER_SCAN,
    IDENTITIES,
    IMAGES_PER_IDENTITY,
    LIVE_SCANS,
    PAPER_DIR,
    RESULTS_DIR,
    ROOT,
    SCANS_DIR,
    SEED,
    ensure_dirs,
)

__version__ = "0.1.0"

#: Submodules whose imports are heavy (torch / matplotlib) and therefore lazy.
_LAZY_MODULES = frozenset({"preprocess", "embed", "plotting"})

__all__ = [
    # constants
    "ROOT",
    "DATA_DIR",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "SCANS_DIR",
    "PAPER_DIR",
    "SEED",
    "FACE_SIZE",
    "EMBED_DIM",
    "IDENTITIES",
    "IMAGES_PER_IDENTITY",
    "LIVE_SCANS",
    "FRAMES_PER_SCAN",
    "ensure_dirs",
    # eager submodules
    "data",
    "maps",
    "metrics",
    # lazily imported submodules
    "preprocess",
    "embed",
    "plotting",
]


def __getattr__(name: str):
    """Lazily import heavy submodules on first attribute access."""
    if name in _LAZY_MODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
