"""Tests for ``fem.preprocess`` — the Olivetti -> model-input recipe.

Why test the exact numbers
--------------------------
The Map B comparison is only valid if the Olivetti embeddings are produced
with the same standardisation the live MTCNN path used.  ``(x - 127.5) / 128``
has exact known outputs for known inputs (0, 0.5, 1), which pins the scale and
the order of the operations without duplicating the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fem.preprocess import standardize, to_model_input

# (0 * 255 - 127.5) / 128 and (1 * 255 - 127.5) / 128
_MIN = -127.5 / 128.0
_MAX = 127.5 / 128.0


def test_to_model_input_shape_dtype_and_constant_values():
    zeros = np.zeros((2, 64, 64), dtype=np.float32)
    output = to_model_input(zeros)

    assert output.shape == (2, 3, 160, 160)
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"  # embedding on MPS is the embedder's job
    np.testing.assert_allclose(output.numpy(), _MIN, atol=1e-6)

    half = np.full((1, 64, 64), 0.5, dtype=np.float32)
    np.testing.assert_allclose(to_model_input(half).numpy(), 0.0, atol=1e-6)

    ones = np.ones((1, 64, 64), dtype=np.float32)
    np.testing.assert_allclose(to_model_input(ones).numpy(), _MAX, atol=1e-6)


def test_to_model_input_replicates_greyscale_to_three_channels():
    rng = np.random.default_rng(0)
    images = rng.random((2, 64, 64), dtype=np.float32)
    output = to_model_input(images).numpy()

    assert np.array_equal(output[:, 0], output[:, 1])
    assert np.array_equal(output[:, 1], output[:, 2])


def test_to_model_input_accepts_channels_first_and_matches_grey_path():
    rng = np.random.default_rng(1)
    grey = rng.random((2, 64, 64), dtype=np.float32)
    rgb = np.repeat(grey[:, None, :, :], 3, axis=1)
    np.testing.assert_array_equal(to_model_input(rgb).numpy(), to_model_input(grey).numpy())


def test_to_model_input_resizes_to_requested_size_and_range():
    rng = np.random.default_rng(2)
    images = rng.random((3, 64, 64), dtype=np.float32)

    assert to_model_input(images, size=80).shape == (3, 3, 80, 80)
    output = to_model_input(images).numpy()
    assert output.min() >= -1.0 and output.max() <= 1.0
    assert np.isfinite(output).all()


def test_to_model_input_rejects_wrong_scale_and_shape():
    uint8_scale = np.full((1, 64, 64), 255.0, dtype=np.float32)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        to_model_input(uint8_scale)

    with pytest.raises(ValueError, match="expected"):
        to_model_input(np.zeros(64, dtype=np.float32))

    with pytest.raises(ValueError, match="channels"):
        to_model_input(np.zeros((1, 2, 64, 64), dtype=np.float32))


def test_standardize_known_values_and_validation():
    tensor = torch.tensor([[0.0, 127.5, 255.0]])
    np.testing.assert_allclose(
        standardize(tensor).numpy(),
        [[-127.5 / 128.0, 0.0, 127.5 / 128.0]],
        atol=1e-7,
    )

    # ndarrays are accepted so callers can avoid importing torch for one cast.
    np.testing.assert_allclose(standardize(np.array([255.0])).numpy(), [_MAX], atol=1e-7)

    with pytest.raises(ValueError):
        standardize(torch.tensor([300.0]))


# ---------------------------------------------------------------------------
# non-finite inputs must fail loudly, not become NaN embeddings
# ---------------------------------------------------------------------------
def test_to_model_input_rejects_nonfinite_values():
    # NaN comparisons are all False, so without an explicit finiteness check a
    # NaN pixel would slip through the min()/max() range validation.
    for bad in (np.nan, np.inf, -np.inf):
        images = np.zeros((1, 64, 64), dtype=np.float32)
        images[0, 0, 0] = bad
        with pytest.raises(ValueError, match="non-finite"):
            to_model_input(images)

    rgb = np.zeros((1, 3, 64, 64), dtype=np.float32)
    rgb[0, 1, 2, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        to_model_input(rgb)


def test_standardize_rejects_nonfinite_values():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite"):
            standardize(torch.tensor([bad]))

    array = np.array([[np.nan, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        standardize(array)
