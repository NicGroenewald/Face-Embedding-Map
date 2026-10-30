"""Shared plumbing for the experiment runners.

Why this module exists
----------------------
Every file in ``results/`` must start with the same header (DESIGN.md §6) so the
paper can cite one provenance block per table.  That header is identical across
experiments, so it is built once here rather than copy-pasted into three
runners (plan trap #4: don't build the pipeline twice).

It also owns the "no NumPy scalars in JSON" rule: ``json.dump`` raises
``TypeError`` on ``np.float32``, and a results file that cannot be reloaded is
worthless.  ``to_jsonable`` walks the object graph once, at write time, instead
of making every caller remember to cast.

Deliberately imports nothing from ``fem`` so that ``analysis/model_arch.py`` can
run before the library exists.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SEED = 0
SCHEMA_VERSION = 1

# Keep in sync with requirements.txt; the paper's reproducibility section quotes
# these exact versions.
_PACKAGES = ("numpy", "scikit-learn", "torch", "facenet-pytorch", "matplotlib")


def _git_commit() -> str:
    """Short HEAD hash, or ``"unknown"`` outside a git checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    """Installed versions via importlib.metadata (facenet-pytorch has no __version__)."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in _PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def common_header(device: str, notes: list[str] | None = None) -> dict[str, Any]:
    """The DESIGN.md §6 provenance block shared by every results file."""
    return {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "device": device,
        "seed": SEED,
        "notes": list(notes or []),
    }


def to_jsonable(obj: Any) -> Any:
    """Recursively convert NumPy/PyTorch scalars and arrays into plain Python.

    Casts ``np.integer`` -> ``int``, ``np.floating`` -> ``float``, arrays/tensors
    -> nested lists, ``Path`` -> ``str``.  Unknown types are left for ``json.dump``
    to reject loudly rather than silently stringified.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    # torch.Tensor: duck-typed to keep this module torch-free.
    if hasattr(obj, "detach"):
        return to_jsonable(obj.detach().cpu().numpy())
    return obj


def save_json(payload: dict[str, Any], path: Path) -> Path:
    """Write ``payload`` as pure JSON, creating parent dirs; return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = to_jsonable(payload)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2, allow_nan=False)
        fh.write("\n")
    return path


def describe_device() -> str:
    """'mps' when Apple's GPU is available, else 'cpu' (never lies about it)."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def save_figure(fig, stem: Path) -> None:
    """Write a matplotlib figure as BOTH figures/<stem>.png (200 dpi) and .pdf.

    Only used for figures that have no ``fem.plotting`` saver (combined scree,
    confusion panel, pooling panel, threshold curves).  Keeps the png+pdf rule in
    one place instead of repeating it per call site.
    """
    import matplotlib.pyplot as plt

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# Colourblind-safe pair used consistently for Map A vs Map B everywhere.
MAP_COLORS = {"Map A": "#0072B2", "Map B": "#D55E00"}


def ensure_importable_root() -> None:
    """Put the repo root on sys.path so ``import fem`` works from analysis/."""
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
