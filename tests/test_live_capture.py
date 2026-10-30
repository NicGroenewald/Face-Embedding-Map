"""Tests for ``scripts/live_capture.py`` that do not need a webcam or an MPS device.

Why test a CLI
--------------
``scripts/live_capture.py`` is the only code path that creates the raw data the live study is
built on.  Its metadata sidecar and "never overwrite" behaviour are easy to
break and impossible to eyeball from a screenshot, so they are tested with
fake cameras and temporary output directories.  The interactive loop itself is
not tested here (it needs a camera); the pure helpers it is assembled from are.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from scripts import live_capture as live
from fem.config import PRIVATE_SCANS_DIR


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def test_parse_args_defaults_and_derived_label():
    args = live.parse_args([])
    assert args.label == "room_front_1"  # derived from intensity/direction/repeat
    assert args.frames == 10
    assert args.repeat == 1
    assert args.warmup == live.DEFAULT_WARMUP
    assert args.outdir == PRIVATE_SCANS_DIR
    assert args.device is None


def test_parse_args_sanitizes_label_and_validates_numbers():
    args = live.parse_args(
        [
            "--label",
            "flash left/3",
            "--intensity",
            "flash",
            "--direction",
            "left",
            "--repeat",
            "3",
            "--frames",
            "5",
            "--warmup",
            "0",
        ]
    )
    assert args.label == "flash-left-3"
    assert args.frames == 5
    assert args.warmup == 0

    with pytest.raises(SystemExit):
        live.parse_args(["--frames", "0"])
    with pytest.raises(SystemExit):
        live.parse_args(["--repeat", "0"])
    with pytest.raises(SystemExit):
        live.parse_args(["--intensity", "moonlight"])


# ---------------------------------------------------------------------------
# detection helpers
# ---------------------------------------------------------------------------
def test_largest_face_picks_largest_box_and_reports_probability():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 20.0, 20.0]])
    probabilities = np.array([0.99, 0.80])
    index, probability = live.largest_face(boxes, probabilities)
    assert index == 1
    assert probability == pytest.approx(0.80)

    assert live.largest_face(None, None) == (None, 0.0)
    assert live.largest_face(np.empty((0, 4)), np.array([])) == (None, 0.0)


class FakeCamera:
    """Minimal ``cv2.VideoCapture`` stand-in: a fixed sequence of frames."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.reads = 0

    def read(self):
        if self.reads >= len(self._frames):
            return False, None
        frame = self._frames[self.reads]
        self.reads += 1
        return True, frame

    def release(self):  # pragma: no cover - exercised in main() only
        pass


def test_warm_up_discards_the_requested_number_of_frames():
    camera = FakeCamera([np.full((4, 4, 3), value, dtype=np.uint8) for value in (0, 10, 100)])
    brightness = live.warm_up(camera, 2)
    assert camera.reads == 2
    assert brightness == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# scan capture (fake camera + fake model, no webcam/MPS needed)
# ---------------------------------------------------------------------------
class FakeMTCNN:
    """Detects one fixed box and returns an all-zero standardised crop."""

    def detect(self, rgb):
        return np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32), np.array([0.99])

    def extract(self, rgb, boxes, save_path=None):
        return torch.zeros((3, live.FACE_SIZE, live.FACE_SIZE), dtype=torch.float32)


class FakeEmbedder:
    """Returns a constant unit vector, like the frozen model but instant."""

    device = "cpu"

    def embed_tensor(self, batch):
        return np.full((batch.shape[0], 512), 1.0 / np.sqrt(512), dtype=np.float32)


def test_capture_scan_returns_a_full_scan():
    frames = [np.full((100, 100, 3), 120, dtype=np.uint8) for _ in range(10)]
    args = live.parse_args(["--frames", "10", "--no-preview", "--label", "scan"])
    result = live.capture_scan(FakeCamera(frames), FakeMTCNN(), FakeEmbedder(), args)

    assert result is not None
    stacked, probabilities = result
    assert stacked.shape == (10, 512)
    assert stacked.dtype == np.float32
    assert probabilities == [0.99] * 10


def test_capture_scan_discards_incomplete_scan():
    # Only four frames available for a ten-frame scan: the scan must not be
    # saved, because the (10, 512) contract would be silently violated.
    frames = [np.full((100, 100, 3), 120, dtype=np.uint8) for _ in range(4)]
    args = live.parse_args(["--frames", "10", "--no-preview", "--label", "scan"])
    assert live.capture_scan(FakeCamera(frames), FakeMTCNN(), FakeEmbedder(), args) is None


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def _unit_frames(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = rng.normal(size=(10, 512)).astype(np.float32)
    return frames / np.linalg.norm(frames, axis=1, keepdims=True)


def test_save_scan_writes_embeddings_and_manifest_compatible_sidecar(tmp_path):
    args = live.parse_args(
        [
            "--label",
            "flash_left_3",
            "--intensity",
            "flash",
            "--direction",
            "left",
            "--repeat",
            "3",
            "--outdir",
            str(tmp_path),
        ]
    )
    stacked = _unit_frames()
    npy_path, json_path = live.save_scan(stacked, [0.95] * 10, args, "mps")

    assert npy_path.exists()
    assert json_path.exists()
    saved = np.load(npy_path)
    np.testing.assert_array_equal(saved, stacked)

    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    assert sidecar["name"] == npy_path.stem
    assert sidecar["label"] == "flash_left_3"
    assert sidecar["file"] == npy_path.name
    assert sidecar["intensity"] == "flash"
    assert sidecar["direction"] == "left"
    assert sidecar["repeat"] == 3
    assert sidecar["frames"] == 10
    assert sidecar["embed_dim"] == 512
    assert sidecar["device"] == "mps"
    assert "InceptionResnetV1" in sidecar["model"]
    assert sidecar["frame_face_probabilities"] == [0.95] * 10


def test_save_scan_never_silently_overwrites(tmp_path):
    args = live.parse_args(["--label", "scan", "--outdir", str(tmp_path)])
    first_npy, first_json = live.save_scan(_unit_frames(0), [0.9] * 10, args, "cpu")
    second_npy, second_json = live.save_scan(_unit_frames(1), [0.9] * 10, args, "cpu")

    assert first_npy != second_npy
    assert json.loads(first_json.read_text())["name"] == first_npy.stem
    assert json.loads(second_json.read_text())["name"] == second_npy.stem
    assert first_npy.exists() and second_npy.exists()
    # The two scans keep their own data.
    assert not np.array_equal(np.load(first_npy), np.load(second_npy))


def test_output_paths_appends_suffix_on_collision(tmp_path):
    npy_path, json_path, stem = live._output_paths(tmp_path, "scan")
    assert stem == "scan"
    npy_path.write_text("x", encoding="utf-8")
    json_path.write_text("y", encoding="utf-8")

    second_npy, second_json, second_stem = live._output_paths(tmp_path, "scan")
    assert second_stem == "scan-2"
    assert second_npy.name == "scan-2.npy"
    assert second_json.name == "scan-2.json"
