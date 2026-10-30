"""Webcam capture CLI for the live lighting study (Stage 5).

What this script does
---------------------
Press ``s`` to capture one scan: ten frames of the same face under one lighting
condition.  Each frame is detected once with MTCNN (on CPU), cropped and
standardised exactly like ``facenet_pytorch`` expects, embedded with the frozen
InceptionResnetV1 (on MPS when available), and the ten unit-norm 512-d vectors
are saved as one ``(10, 512)`` ``.npy`` file plus a JSON sidecar that records
what the scan *means* (intensity, direction, repeat, timestamp, model, device).

Why the details look the way they do
------------------------------------
The original version of this script lost data in three Mac-specific ways, so
each is handled explicitly here:

* **MTCNN cannot run on MPS on this machine** (adaptive-pool unsupported).  The
  detector is pinned to CPU while the embedder stays on MPS; mixing is free
  because only the 160x160 crop crosses between them.
* **The first webcam frames are black** while auto-exposure settles (measured
  mean brightness 3/255 vs ~100 after).  ``--warmup`` frames are discarded
  before the interactive loop.
* **MPS + webcam windows need a clean shutdown**, otherwise the camera stays
  claimed: everything runs under ``try/finally`` and windows are destroyed.

It also fixes the old detection bug: the previous script called ``mtcnn()``
(which detects internally) *and then* ``mtcnn.detect()`` again, drew the second
detection's box, and embedded the first detection's crop.  Here there is one
detect call per frame, the largest box is chosen from that call, and that same
box is used for both the HUD rectangle and the crop (``mtcnn.extract``), so what
you see is what gets embedded.

Run it
------
    export MPLCONFIGDIR="$PWD/.cache/mpl"
    /Users/nic/.venvs/nn-crew/bin/python scripts/live_capture.py --label flash_left_3 \\
        --intensity flash --direction left --repeat 3

Keys in the preview window: ``s`` = scan, ``q``/``Esc`` = quit.  ``--no-preview``
captures exactly one scan and exits (useful on a machine without a display).

The resulting ``.json`` sidecar contains the fields needed to add the scan to
``data/scans.json``, which is the manifest ``fem.data.load_scans`` reads; the
script does not edit the manifest itself, because the historical 12-scan
manifest is curated and tests pin its size.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

# Allow `python scripts/live_capture.py` from the repo root (the script's own
# directory is on sys.path, the repo root is not). Importing it as
# `scripts.live_capture` is unaffected.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from fem.config import FACE_SIZE, SCANS_DIR
from fem.embed import MODEL_NAME, FaceEmbedder

WINDOW_TITLE = "Face Embedding Map - webcam"
MIN_FACE_PROBABILITY = 0.90
DEFAULT_FRAMES = 10
DEFAULT_WARMUP = 10
INTENSITIES = ("room", "flash", "other")
DIRECTIONS = ("front", "left", "right", "bottom")


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def _sanitize_label(label: str) -> str:
    """Make a label safe to use in a filename, keeping it human-readable."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-._")
    return cleaned or "scan"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments; the label defaults to ``<intensity>_<direction>_<repeat>``."""
    parser = argparse.ArgumentParser(
        description="Capture one scan (frames x 512-d embeddings) from the webcam."
    )
    parser.add_argument(
        "--label",
        default=None,
        help="scan name (default: <intensity>_<direction>_<repeat>)",
    )
    parser.add_argument(
        "--intensity", choices=INTENSITIES, default="room", help="light source (default: room)"
    )
    parser.add_argument(
        "--direction",
        choices=DIRECTIONS,
        default="front",
        help="light direction (default: front)",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="repeat index for this condition (default: 1)"
    )
    parser.add_argument(
        "--frames", type=int, default=DEFAULT_FRAMES, help=f"frames per scan (default: {DEFAULT_FRAMES})"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=SCANS_DIR,
        help="output directory (default: data/scans/)",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"frames discarded while auto-exposure settles (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="embedder device (default: mps if available, else cpu)",
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="capture one scan with no windows"
    )

    args = parser.parse_args(argv)
    if args.frames < 1:
        parser.error("--frames must be >= 1")
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    args.label = _sanitize_label(args.label or f"{args.intensity}_{args.direction}_{args.repeat}")
    return args


# ---------------------------------------------------------------------------
# camera + detection helpers
# ---------------------------------------------------------------------------
def open_camera(index: int) -> cv2.VideoCapture:
    """Open a webcam and fail loudly if the index is wrong/claimed."""
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"could not open camera index {index}; close other apps using it or pass --camera"
        )
    return camera


def warm_up(camera: cv2.VideoCapture, n_frames: int) -> float:
    """Discard the dark startup frames; return the last discarded frame's brightness.

    The webcam's auto-exposure needs a moment.  Reading a handful of frames
    before showing anything avoids both the "no face on frame 1" failure and
    storing a black frame in a scan.
    """
    brightness = 0.0
    discarded = 0
    for _ in range(n_frames):
        ok, frame = camera.read()
        if not ok:
            continue
        discarded += 1
        brightness = float(frame.mean())
    print(f"[live] warm-up: discarded {discarded} frame(s), last brightness {brightness:.1f}/255")
    return brightness


def largest_face(
    boxes: np.ndarray | None, probabilities: np.ndarray | None
) -> tuple[int | None, float]:
    """Index and probability of the largest detected face box (``None`` if none)."""
    if boxes is None or len(boxes) == 0:
        return None, 0.0
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    index = int(np.argmax(areas))
    probability = float(probabilities[index]) if probabilities is not None else 0.0
    return index, probability


def extract_face(mtcnn: MTCNN, rgb: np.ndarray, box: np.ndarray) -> torch.Tensor | None:
    """Crop + standardise one face using the box from a single detect call.

    ``MTCNN.extract`` is the exact code path ``MTCNN.forward`` uses, so the
    resulting tensor is byte-for-byte what ``mtcnn(rgb)`` would produce
    (verified: max difference 0.0).  Passing the selected box explicitly is
    what lets detection happen only once per frame.
    """
    crop = mtcnn.extract(rgb, np.asarray(box, dtype=np.float32).reshape(1, 4), None)
    if crop is None:
        return None
    if crop.shape != (3, FACE_SIZE, FACE_SIZE):
        raise RuntimeError(f"unexpected crop shape {tuple(crop.shape)}")
    return crop


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
def _put_text(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.6) -> None:
    """White text with a black outline, readable on any webcam background."""
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    args: argparse.Namespace,
    device: str,
    boxes: np.ndarray | None = None,
    best_index: int | None = None,
    status: str = "",
) -> None:
    """Draw detection boxes and the key/status HUD in place."""
    if boxes is not None:
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = (int(value) for value in box[:4])
            if index == best_index:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    lines = [
        status or "Press 's' to scan",
        f"{args.label}  [{args.intensity}/{args.direction}, repeat {args.repeat}]",
        f"device {device} - MTCNN cpu - S scan {args.frames} frames - Q quit",
    ]
    for offset, line in enumerate(lines):
        _put_text(frame, line, (12, 30 + 28 * offset))


# ---------------------------------------------------------------------------
# scan capture + persistence
# ---------------------------------------------------------------------------
def capture_scan(
    camera: cv2.VideoCapture,
    mtcnn: MTCNN,
    embedder: FaceEmbedder,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[float]] | None:
    """Capture ``args.frames`` usable frames; ``None`` if the scan is incomplete.

    A scan is saved only when every requested frame contains a detectable face.
    A short scan would silently violate the ``(10, 512)`` contract the analysis
    and the manifest rely on, so the safer behaviour is to discard it and let
    the user retry.
    """
    embeddings: list[np.ndarray] = []
    probabilities: list[float] = []

    for frame_index in range(args.frames):
        ok, frame = camera.read()
        if not ok:
            print("[live] camera read failed mid-scan; discarding scan")
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Exactly one detection per frame; the chosen box is reused for the
        # crop below and drawn on screen.
        boxes, all_probabilities = mtcnn.detect(rgb)
        best_index, probability = largest_face(boxes, all_probabilities)

        if best_index is None or probability < MIN_FACE_PROBABILITY:
            status = f"no face above {MIN_FACE_PROBABILITY:.2f} (best {probability:.3f}) - skipping"
            print(f"[live] frame {frame_index + 1}/{args.frames}: {status}")
        else:
            crop = extract_face(mtcnn, rgb, boxes[best_index])
            if crop is None:
                status = "crop failed - skipping"
                print(f"[live] frame {frame_index + 1}/{args.frames}: {status}")
            else:
                embedding = embedder.embed_tensor(crop.unsqueeze(0))[0]
                embeddings.append(embedding)
                probabilities.append(probability)
                status = f"captured {len(embeddings)}/{args.frames}"
                print(f"[live] frame {frame_index + 1}/{args.frames}: {status}")

        if not args.no_preview:
            display = frame.copy()
            draw_hud(display, args, embedder.device, boxes, best_index, status)
            cv2.imshow(WINDOW_TITLE, display)
            cv2.waitKey(1)

    if len(embeddings) != args.frames:
        print(
            f"[live] scan incomplete: {len(embeddings)}/{args.frames} frames usable; "
            "nothing saved (retry when the face is lit and centred)"
        )
        return None

    return np.stack(embeddings).astype(np.float32, copy=False), probabilities


def _output_paths(outdir: Path, stem: str) -> tuple[Path, Path, str]:
    """Return unique ``.npy``/``.json`` paths; never overwrite an earlier scan."""
    outdir.mkdir(parents=True, exist_ok=True)
    candidate = stem
    suffix = 2
    while True:
        npy_path = outdir / f"{candidate}.npy"
        json_path = outdir / f"{candidate}.json"
        if not npy_path.exists() and not json_path.exists():
            return npy_path, json_path, candidate
        candidate = f"{stem}-{suffix}"
        suffix += 1


def save_scan(
    stacked: np.ndarray,
    probabilities: list[float],
    args: argparse.Namespace,
    device: str,
) -> tuple[Path, Path]:
    """Write the ``(frames, 512)`` embeddings and a manifest-compatible sidecar."""
    stamp = dt.datetime.now().strftime("%d%m%Y-%H%M%S")
    npy_path, json_path, name = _output_paths(Path(args.outdir), f"{stamp}-{args.label}")
    np.save(npy_path, stacked)

    metadata = {
        "format_version": 1,
        "name": args.label,
        "file": npy_path.name,
        "intensity": args.intensity,
        "direction": args.direction,
        "repeat": args.repeat,
        "frames": int(stacked.shape[0]),
        "embed_dim": int(stacked.shape[1]),
        "captured": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "device": device,
        "detector": "MTCNN(cpu)",
        "min_face_probability": MIN_FACE_PROBABILITY,
        "frame_face_probabilities": [float(value) for value in probabilities],
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return npy_path, json_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Interactive capture loop; returns a process exit code."""
    args = parse_args(argv)

    # MTCNN on CPU, embedder on MPS when available: see module docstring.
    mtcnn = MTCNN(image_size=FACE_SIZE, device="cpu")
    embedder = FaceEmbedder(device=args.device)
    print(f"[live] embedder: {embedder}")

    try:
        camera = open_camera(args.camera)
    except RuntimeError as error:
        print(f"[live] {error}")
        return 1

    saved = 0
    try:
        warm_up(camera, args.warmup)
        print(
            "[live] controls: 's' = scan, 'q'/Esc = quit. "
            f"output: {Path(args.outdir).resolve()}"
        )
        while True:
            ok, frame = camera.read()
            if not ok:
                print("[live] camera read failed; stopping")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, all_probabilities = mtcnn.detect(rgb)
            best_index, probability = largest_face(boxes, all_probabilities)
            status = (
                "face detected - press 's'"
                if best_index is not None and probability >= MIN_FACE_PROBABILITY
                else "no face detected"
            )
            display = frame.copy()
            draw_hud(display, args, embedder.device, boxes, best_index, status)
            if not args.no_preview:
                cv2.imshow(WINDOW_TITLE, display)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = ord("s")  # automated single scan

            if key in (ord("q"), 27):
                break
            if key != ord("s"):
                continue

            result = capture_scan(camera, mtcnn, embedder, args)
            if result is not None:
                stacked, probabilities = result
                npy_path, json_path = save_scan(stacked, probabilities, args, embedder.device)
                saved += 1
                print(f"[live] saved {stacked.shape} -> {npy_path}")
                print(f"[live] metadata -> {json_path}")

            if args.no_preview:
                break

        print(f"[live] done: {saved} scan(s) saved")
        if args.no_preview and saved == 0:
            # Scripted capture is only useful if it produced a scan; make the
            # failure visible to whatever launched it.
            print("[live] no scan captured in --no-preview mode", file=sys.stderr)
            return 1
        return 0
    finally:
        # Always release the camera and close windows: a crashed preview loop
        # otherwise leaves the device claimed until the process exits.
        camera.release()
        if not args.no_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
