"""Load the two datasets the project measures on.

There are exactly two data sources and they must be loaded the *same way every
time* — Map A and Map B are compared on identical images, so a second bespoke
loader would invalidate the comparison (plan trap #4: don't build the pipeline
twice).

Map A / Map B (Olivetti)
    ``load_olivetti`` wraps scikit-learn's cached download and normalises the
    return type: float32 images in [0, 1] and int64 labels.  It deliberately
    does not flatten the images — the caller decides whether it wants raw
    pixels (``reshape(n, -1)``) or model input (``preprocess.to_model_input``).

Live study (12 webcam scans)
    ``load_scans`` reads ``data/scans.json``.  The filenames alone do not say
    what a scan *means* ("06092026-light_face_left_1.npy" encodes the capture
    date, not a condition name), so the manifest is the source of truth for
    intensity/direction/repeat metadata.  Fixing a mislabelled condition is
    then a manifest edit, not a code edit.

Both loaders validate the data they hand out.  A shape or unit-norm check here
turns a silent, wrong result in the paper into a loud exception at load time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import (
    EMBED_DIM,
    FRAMES_PER_SCAN,
    IMAGES_PER_IDENTITY,
    IDENTITIES,
    OLIVETTI_HOME,
    OLIVETTI_IMAGE_SHAPE,
    SCANS_DIR,
    SCANS_MANIFEST,
)


def load_olivetti(
    data_home: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images, labels)`` for the Olivetti faces.

    Parameters
    ----------
    data_home:
        Directory holding scikit-learn's ``olivetti_py3.pkz`` cache.  Defaults
        to ``data/sklearn`` inside the repo so the normal path never touches
        the network.

    Returns
    -------
    images:
        float32 array ``(400, 64, 64)`` with values in ``[0, 1]``, exactly as
        scikit-learn stores them.
    labels:
        int64 array ``(400,)`` in ``0..39``; ten consecutive images per label.
    """
    # Imported here rather than at module scope: sklearn is already a hard
    # dependency of the project, but this keeps ``import fem.data`` cheap for
    # the manifest-only paths (e.g. scripts/compare_scans.py never needs it).
    from sklearn.datasets import fetch_olivetti_faces

    home = Path(data_home) if data_home is not None else OLIVETTI_HOME
    home.mkdir(parents=True, exist_ok=True)

    try:
        # Explicit offline attempt first: if the cache exists this is a pure
        # local read.  Only a genuinely missing cache is allowed to download.
        dataset = fetch_olivetti_faces(data_home=str(home), download_if_missing=False)
    except OSError:
        dataset = fetch_olivetti_faces(data_home=str(home))

    images = np.asarray(dataset.images, dtype=np.float32)
    labels = np.asarray(dataset.target, dtype=np.int64)

    expected = (IDENTITIES * IMAGES_PER_IDENTITY,) + OLIVETTI_IMAGE_SHAPE
    if images.shape != expected:
        raise ValueError(f"unexpected Olivetti image shape {images.shape}, expected {expected}")
    if labels.shape != (IDENTITIES * IMAGES_PER_IDENTITY,):
        raise ValueError(f"unexpected Olivetti label shape {labels.shape}")
    if labels.min() < 0 or labels.max() != IDENTITIES - 1:
        raise ValueError(f"unexpected Olivetti label range [{labels.min()}, {labels.max()}]")
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size != IDENTITIES or not np.all(counts == IMAGES_PER_IDENTITY):
        raise ValueError("Olivetti is expected to have 40 identities x 10 images each")

    return images.copy(), labels.copy()


@dataclass(frozen=True)
class Scan:
    """One webcam scan: a short burst of frames under one lighting condition.

    ``frames`` is ``(10, 512)`` float32 with *already unit-norm* rows.
    facenet-pytorch's ``InceptionResnetV1.forward`` ends in ``F.normalize``, so
    the distance between any two rows is bounded by 2 and the cosine
    similarity is exactly ``1 - d**2 / 2`` (see ``metrics.cosine_from_euclidean``).
    """

    name: str
    path: Path
    intensity: str  # "room" | "flash"
    direction: str  # "front" | "left" | "right" | "bottom"
    frames: np.ndarray  # (FRAMES_PER_SCAN, EMBED_DIM) float32, unit-norm rows

    @property
    def condition(self) -> str:
        """Human/plot label for the lighting condition, e.g. ``"flash/left"``."""
        return f"{self.intensity}/{self.direction}"

    @property
    def repeat(self) -> int:
        """Repeat index parsed from the manifest name (``..._1`` -> 1)."""
        suffix = self.name.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0


@dataclass
class ScanSet:
    """The 12 live scans as one object, with the aggregations the study needs.

    The methods here exist because every runner needs the same reshaping of
    the raw ``list[Scan]`` into arrays.  Centralising them keeps the within-scan
    jitter, the between-scan distances and the pooled embeddings computed by
    one piece of code (plan trap #4).
    """

    scans: list[Scan]
    subject: str = ""
    captured: str = ""
    frames_per_scan: int = FRAMES_PER_SCAN
    manifest: dict = field(default_factory=dict)

    # -- basic access -------------------------------------------------------
    def __len__(self) -> int:
        return len(self.scans)

    def __iter__(self):
        return iter(self.scans)

    def __getitem__(self, index: int) -> Scan:
        return self.scans[index]

    def names(self) -> list[str]:
        """Scan names in manifest order."""
        return [scan.name for scan in self.scans]

    def by_name(self, name: str) -> Scan:
        """Look up a scan by name; raises ``KeyError`` if absent."""
        for scan in self.scans:
            if scan.name == name:
                return scan
        raise KeyError(f"no scan named {name!r}; available: {self.names()}")

    # -- array shapes -------------------------------------------------------
    def stack(self) -> np.ndarray:
        """All frames as ``(n_scans, frames_per_scan, 512)``.

        Returns a fresh array so callers cannot mutate the loaded scans.
        """
        if not self.scans:
            raise ValueError("empty ScanSet")
        shapes = {scan.frames.shape for scan in self.scans}
        if len(shapes) != 1:
            raise ValueError(f"scans have inconsistent frame shapes: {sorted(shapes)}")
        return np.stack([scan.frames for scan in self.scans])

    def frames(self) -> np.ndarray:
        """Every frame as ``(n_scans * frames_per_scan, 512)``, scan order."""
        return self.stack().reshape(-1, EMBED_DIM)

    def mean_embeddings(self) -> np.ndarray:
        """Scan means as ``(n_scans, 512)``, re-normalised to unit length.

        Averaging unit vectors shortens them; re-normalising keeps the
        unit-vector distance/cosine relation valid for pooled comparisons
        (the "average N frames at enrolment" recommendation in E3).
        """
        means = self.stack().mean(axis=1)
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("a scan mean is the zero vector; cannot normalise")
        return (means / norms).astype(np.float32, copy=False)

    def labels(self) -> np.ndarray:
        """``(n_scans,)`` condition strings such as ``"flash/left"``."""
        return np.array([scan.condition for scan in self.scans], dtype=str)

    # -- grouping helpers used by the lighting study ------------------------
    def conditions(self) -> list[str]:
        """Unique condition strings in first-seen manifest order."""
        seen: list[str] = []
        for scan in self.scans:
            if scan.condition not in seen:
                seen.append(scan.condition)
        return seen

    def by_condition(self) -> dict[str, list[Scan]]:
        """Group scans by ``condition``, preserving manifest order."""
        groups: dict[str, list[Scan]] = {}
        for scan in self.scans:
            groups.setdefault(scan.condition, []).append(scan)
        return groups


def _resolve_scan_path(file: str) -> Path:
    """Manifest ``file`` entries are relative to ``data/scans/``."""
    path = Path(file)
    return path if path.is_absolute() else SCANS_DIR / path


def load_scans(manifest: str | Path | None = None) -> ScanSet:
    """Load the live study scans described by a manifest.

    Parameters
    ----------
    manifest:
        Path to a JSON manifest.  Defaults to ``data/scans.json``.

    The manifest is the source of truth for metadata: the raw filenames encode
    a capture timestamp, not a lighting condition.  Rows must already be unit
    norm — facenet-pytorch normalises its output, and the whole
    Euclidean-to-cosine conversion depends on that invariant, so a violated
    invariant is an error rather than something silently re-normalised.
    """
    manifest_path = Path(manifest) if manifest is not None else SCANS_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"scan manifest not found: {manifest_path}. "
            "The manifest is checked in at data/scans.json."
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("scans")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{manifest_path}: 'scans' must be a non-empty list")

    frames_per_scan = int(payload.get("frames_per_scan", FRAMES_PER_SCAN))
    scans: list[Scan] = []

    for entry in entries:
        try:
            name = str(entry["name"])
            file = str(entry["file"])
            intensity = str(entry["intensity"])
            direction = str(entry["direction"])
        except KeyError as exc:
            raise ValueError(f"{manifest_path}: scan entry missing key {exc}") from None

        path = _resolve_scan_path(file)
        if not path.exists():
            raise FileNotFoundError(f"scan {name!r} points at missing file {path}")

        frames = np.load(path).astype(np.float32, copy=False)
        if frames.ndim != 2 or frames.shape != (frames_per_scan, EMBED_DIM):
            raise ValueError(
                f"scan {name!r} has shape {frames.shape}, "
                f"expected ({frames_per_scan}, {EMBED_DIM})"
            )
        if not np.isfinite(frames).all():
            raise ValueError(f"scan {name!r} contains non-finite values")

        norms = np.linalg.norm(frames, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError(
                f"scan {name!r} rows are not unit-norm "
                f"(min={norms.min():.6f}, max={norms.max():.6f}); "
                "the live pipeline is expected to store L2-normalised embeddings"
            )

        scans.append(
            Scan(
                name=name,
                path=path,
                intensity=intensity,
                direction=direction,
                frames=frames,
            )
        )

    return ScanSet(
        scans=scans,
        subject=str(payload.get("subject", "")),
        captured=str(payload.get("captured", "")),
        frames_per_scan=frames_per_scan,
        manifest=payload,
    )
