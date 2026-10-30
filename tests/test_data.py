"""Tests for ``fem.data``: Olivetti loading and the scan manifest.

Why a manifest integrity test
-----------------------------
The live study's condition labels ("room/front", "flash/left", ...) exist only
in ``data/scans.json``; the ``.npy`` filenames encode a timestamp.  A swapped
or mistyped entry would silently attribute an effect to the wrong lighting
condition, so the test checks the manifest's shape, uniqueness and counts, not
just that loading does not crash.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from fem import config, data

OLIVETTI_CACHE = config.DATA_DIR / "sklearn" / "olivetti_py3.pkz"

EXPECTED_CONDITION_COUNTS = {
    "room/front": 2,
    "room/left": 2,
    "flash/front": 2,
    "flash/left": 2,
    "flash/right": 2,
    "flash/bottom": 2,
}


# ---------------------------------------------------------------------------
# Olivetti
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not OLIVETTI_CACHE.exists(),
    reason="Olivetti cache missing; call fem.data.load_olivetti() once with network",
)
def test_load_olivetti_shape_dtype_and_labels():
    images, labels = data.load_olivetti()

    assert images.shape == (400, 64, 64)
    assert images.dtype == np.float32
    assert images.min() >= 0.0 and images.max() <= 1.0
    assert labels.shape == (400,)
    assert labels.dtype == np.int64
    assert set(np.unique(labels)) == set(range(40))
    counts = Counter(labels.tolist())
    assert all(count == 10 for count in counts.values())


# ---------------------------------------------------------------------------
# live scans
# ---------------------------------------------------------------------------
def test_manifest_integrity():
    payload = json.loads(config.SCANS_MANIFEST.read_text(encoding="utf-8"))

    assert payload["format_version"] == 1
    assert payload["frames_per_scan"] == 10
    assert payload["subject"] == "nic"
    assert payload["captured"] == "2026-09-06"
    entries = payload["scans"]
    assert len(entries) == config.LIVE_SCANS

    names = [entry["name"] for entry in entries]
    files = [entry["file"] for entry in entries]
    assert len(set(names)) == len(names)
    assert len(set(files)) == len(files)

    for entry in entries:
        assert set(entry) == {"name", "file", "intensity", "direction", "repeat"}
        assert entry["intensity"] in {"room", "flash"}
        assert entry["direction"] in {"front", "left", "right", "bottom"}
        assert entry["repeat"] in {1, 2}

        path = config.SCANS_DIR / entry["file"]
        assert path.exists(), f"manifest points at missing file {path}"
        frames = np.load(path)
        assert frames.shape == (config.FRAMES_PER_SCAN, config.EMBED_DIM)
        assert frames.dtype == np.float32
        assert np.allclose(np.linalg.norm(frames, axis=1), 1.0, atol=1e-5)


def test_load_scans_shapes_conditions_and_norms():
    scan_set = data.load_scans()

    assert len(scan_set) == config.LIVE_SCANS
    assert scan_set.stack().shape == (12, 10, 512)
    assert scan_set.frames().shape == (120, 512)
    assert scan_set.mean_embeddings().shape == (12, 512)

    assert np.allclose(np.linalg.norm(scan_set.frames(), axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(scan_set.mean_embeddings(), axis=1), 1.0, atol=1e-5)

    assert Counter(scan_set.labels().tolist()) == EXPECTED_CONDITION_COUNTS
    assert scan_set.subject == "nic"
    assert scan_set.captured == "2026-09-06"
    assert scan_set.conditions() == [
        "room/front",
        "room/left",
        "flash/front",
        "flash/left",
        "flash/right",
        "flash/bottom",
    ]
    assert [scan.repeat for scan in scan_set] == [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

    flash_left_2 = scan_set.by_name("flash_left_2")
    assert flash_left_2.condition == "flash/left"
    assert flash_left_2.repeat == 2
    assert flash_left_2.path.name == "06092026-phone_light_face_left_2.npy"


def test_each_scan_frame_matches_its_file():
    scan_set = data.load_scans()
    for scan in scan_set:
        np.testing.assert_array_equal(scan.frames, np.load(scan.path))

    with pytest.raises(KeyError):
        scan_set.by_name("does_not_exist")


def test_load_scans_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_scans(tmp_path / "no-such-manifest.json")


def test_load_scans_missing_scan_file_raises(tmp_path):
    manifest = {
        "format_version": 1,
        "subject": "test",
        "captured": "2026-01-01",
        "frames_per_scan": 10,
        "scans": [
            {
                "name": "room_front_1",
                "file": "missing.npy",
                "intensity": "room",
                "direction": "front",
                "repeat": 1,
            }
        ],
    }
    manifest_path = tmp_path / "scans.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing file"):
        data.load_scans(manifest_path)


def test_load_scans_rejects_non_unit_norm_rows(tmp_path):
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(10, 512)).astype(np.float32)  # norms ~22, not unit
    scan_path = tmp_path / "bad.npy"
    np.save(scan_path, frames)

    manifest = {
        "format_version": 1,
        "frames_per_scan": 10,
        "scans": [
            {
                "name": "bad_scan",
                "file": str(scan_path),  # absolute path is resolved as-is
                "intensity": "room",
                "direction": "front",
                "repeat": 1,
            }
        ],
    }
    manifest_path = tmp_path / "scans.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unit-norm"):
        data.load_scans(manifest_path)
