"""Turn Olivetti negatives into exactly the pixels InceptionResnetV1 expects.

Why this module exists
----------------------
The live embeddings (Map B on webcam frames) were produced by
``facenet_pytorch``'s MTCNN path:

    BGR frame -> RGB -> crop the face box -> cv2.resize to 160x160
    -> uint8 pixels 0..255 -> (x - 127.5) / 128

For the Olivetti comparison to be honest, Map B's Olivetti embeddings must be
fed the same *kind* of tensor: 3-channel, 160x160, same fixed mean/std.  The
only difference is the shortcut around the detector: Olivetti is already
pre-aligned, so there is no face box to find — we resize the whole 64x64
image instead of cropping it.  That shortcut is a documented limitation of the
study, but the standardisation is not negotiable: a model fed the wrong scale
would produce embeddings that are not comparable with the live ones.

``fixed_image_standardization`` is imported from facenet-pytorch itself
instead of re-implementing ``(x - 127.5) / 128``.  If upstream ever changes
the constant, the project should follow it rather than copy it.

This module is torch-only (no torchvision, no matplotlib) so it is cheap to
import from a test.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch.models.mtcnn import fixed_image_standardization

from .config import FACE_SIZE


def standardize(batch_uint8: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Apply facenet-pytorch's fixed standardisation to 0..255 pixels.

    ``(x - 127.5) / 128`` maps ``0..255`` to ``-0.99609..0.99609``.  This is
    the exact function MTCNN applies when ``post_process=True``, which is how
    the live embeddings were produced.

    Parameters
    ----------
    batch_uint8:
        Tensor or array of pixel values on the 0..255 scale (any shape).

    Returns
    -------
    float32 tensor with the same shape, standardised.
    """
    tensor = torch.as_tensor(batch_uint8, dtype=torch.float32)
    if tensor.numel() and not torch.isfinite(tensor).all():
        raise ValueError(
            "standardize() input contains non-finite values (NaN or infinity); "
            "these would propagate into the model embeddings"
        )
    if tensor.numel() and (tensor.min() < -1e-4 or tensor.max() > 255.0001):
        raise ValueError(
            "standardize() expects pixels on the 0..255 scale; "
            f"got range [{tensor.min().item():.4f}, {tensor.max().item():.4f}]"
        )
    return fixed_image_standardization(tensor)


def to_model_input(images: np.ndarray, size: int = FACE_SIZE) -> torch.Tensor:
    """Convert ``(N, 64, 64)`` Olivetti images to model input.

    Recipe (must match the live MTCNN path's *scale*, not its cropping):

    1. replicate greyscale to 3 channels,
    2. bilinear resize to ``(size, size)`` with ``antialias=True``,
    3. scale to 0..255,
    4. ``(x - 127.5) / 128``.

    Parameters
    ----------
    images:
        ``(N, 64, 64)`` float images in ``[0, 1]`` (Olivetti's native range),
        or already ``(N, 3, H, W)`` float images in ``[0, 1]``.
    size:
        Target edge length; 160 for InceptionResnetV1.

    Returns
    -------
    CPU float32 tensor ``(N, 3, size, size)``.  Moving it to MPS is the
    embedder's job.
    """
    array = np.ascontiguousarray(np.asarray(images, dtype=np.float32))
    # Check finiteness *before* the range test: NaN comparisons are all False,
    # so a NaN image would otherwise slip through min()/max() and become a
    # NaN embedding.
    if not np.isfinite(array).all():
        raise ValueError(
            "to_model_input() input contains non-finite values (NaN or infinity); "
            "these would propagate into the model embeddings"
        )
    if array.ndim == 3:
        tensor = torch.from_numpy(array).unsqueeze(1)  # (N, 1, H, W)
    elif array.ndim == 4:
        tensor = torch.from_numpy(array)
    else:
        raise ValueError(
            f"expected (N, H, W) or (N, C, H, W) images, got shape {array.shape}"
        )

    if tensor.numel() == 0:
        raise ValueError("empty image batch")

    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    elif tensor.shape[1] != 3:
        raise ValueError(
            f"expected 1 or 3 channels, got {tensor.shape[1]} (shape {tuple(tensor.shape)})"
        )

    if tensor.min() < -1e-4 or tensor.max() > 1.0001:
        raise ValueError(
            "to_model_input() expects float images in [0, 1] "
            f"(Olivetti's native range); got [{tensor.min().item():.4f}, "
            f"{tensor.max().item():.4f}]. Did you pass 0..255 pixels?"
        )

    if tuple(tensor.shape[-2:]) != (size, size):
        tensor = F.interpolate(
            tensor,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    # 0..1 float -> 0..255 float, then the model's fixed standardisation.
    return standardize(tensor * 255.0).contiguous()
