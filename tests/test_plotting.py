"""Tests for ``fem.plotting``: every saver must write PNG *and* vector PDF.

Why test the output contract instead of rendering
-------------------------------------------------
The paper consumes PDFs and the README consumes PNGs; a saver that silently
writes only one of them is a paper-day surprise.  These tests call each saver
once with small synthetic data and check the files exist, are non-empty, have
the expected magic bytes, and that the PNG really is 200 dpi.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from fem import plotting


@pytest.fixture()
def output_dir(tmp_path):
    return tmp_path


def assert_png_and_pdf(base_path) -> None:
    """A saver must produce <base>.png at 200 dpi and <base>.pdf."""
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    assert png_path.exists(), f"missing {png_path}"
    assert pdf_path.exists(), f"missing {pdf_path}"
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0
    assert png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert pdf_path.read_bytes()[:5] == b"%PDF-"

    with Image.open(png_path) as image:
        dpi = image.info.get("dpi", (0, 0))
    assert float(dpi[0]) == pytest.approx(200.0, abs=1.0)


def test_save_scatter(output_dir):
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(20, 2))
    labels = np.repeat(np.arange(4), 5)
    # Passing an explicit suffix must still write both formats.
    plotting.save_scatter(
        coords,
        labels,
        output_dir / "scatter.png",
        title="test map",
        identity_names=["a", "b", "c", "d"],
    )
    assert_png_and_pdf(output_dir / "scatter")


def test_save_scatter_validates():
    with pytest.raises(ValueError):
        plotting.save_scatter(np.zeros((5, 3)), np.zeros(5), "unused")


def test_save_eigenfaces(output_dir):
    rng = np.random.default_rng(0)
    plotting.save_eigenfaces(rng.normal(size=(16, 4096)), output_dir / "eigenfaces")
    assert_png_and_pdf(output_dir / "eigenfaces")


def test_save_scree_single_and_two_maps(output_dir):
    rng = np.random.default_rng(0)
    evr = np.sort(rng.random(30))[::-1]
    evr = evr / evr.sum()
    plotting.save_scree(evr, output_dir / "scree_single")
    assert_png_and_pdf(output_dir / "scree_single")

    plotting.save_scree({"Map A": evr, "Map B": evr * 0.9}, output_dir / "scree_both")
    assert_png_and_pdf(output_dir / "scree_both")


def test_save_distance_histograms(output_dir):
    rng = np.random.default_rng(0)
    plt_data = {
        "Map A": (rng.normal(0.6, 0.1, 200), rng.normal(0.9, 0.1, 500)),
        "Map B": (rng.normal(0.5, 0.05, 200), rng.normal(0.8, 0.1, 500)),
    }
    plotting.save_distance_histograms(plt_data, output_dir / "histograms")
    assert_png_and_pdf(output_dir / "histograms")


def test_save_roc_and_threshold_sweep(output_dir):
    rng = np.random.default_rng(0)
    curves = {
        "Map A": (rng.normal(8.4, 0.5, 200), rng.normal(12.4, 2.0, 800)),
        "Map B": (rng.normal(0.8, 0.05, 200), rng.normal(1.4, 0.25, 800)),
    }
    plotting.save_roc(curves, output_dir / "roc")
    assert_png_and_pdf(output_dir / "roc")

    plotting.save_threshold_sweep(curves, output_dir / "threshold_overlay")
    assert_png_and_pdf(output_dir / "threshold_overlay")

    # The maps' distance scales differ by 10x here, so panels are the
    # readable form; both must be written the same way.
    plotting.save_threshold_sweep(
        curves, output_dir / "threshold_panels", separate_panels=True
    )
    assert_png_and_pdf(output_dir / "threshold_panels")


def test_save_heatmap(output_dir):
    rng = np.random.default_rng(0)
    matrix = rng.random((12, 12))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0.0)
    labels = [f"scan_{index}" for index in range(12)]
    plotting.save_heatmap(matrix, labels, output_dir / "heatmap", title="scan distances")
    assert_png_and_pdf(output_dir / "heatmap")

    with pytest.raises(ValueError):
        plotting.save_heatmap(matrix, labels[:5], output_dir / "bad")


def test_save_metric_bars_both_accepted_shapes(output_dir):
    plotting.save_metric_bars(
        {"silhouette": (0.105, 0.497), "auc": (0.881, 0.9995)},
        output_dir / "bars_pairs",
        ylabel="score",
    )
    assert_png_and_pdf(output_dir / "bars_pairs")

    plotting.save_metric_bars(
        {"Map A": {"silhouette": 0.105, "auc": 0.881}, "Map B": {"silhouette": 0.497, "auc": 0.9995}},
        output_dir / "bars_nested",
    )
    assert_png_and_pdf(output_dir / "bars_nested")


def test_save_lighting_panel(output_dir):
    live = {
        "within_scan_jitter": [
            {"scan": f"s{index}", "min": 0.18, "mean": 0.28, "max": 0.37} for index in range(12)
        ],
        "same_condition": [
            {"condition": f"c{index}", "min": 0.2, "mean": 0.3, "max": 0.4} for index in range(6)
        ],
        "intensity_effect": {"condition_pair": ["room/front", "flash/front"], "min": 0.2, "mean": 0.3, "max": 0.38},
        "direction_effect": {
            "pairs": [{"pair": ["flash/front", "flash/left"], "min": 0.63, "mean": 0.72, "max": 0.8}],
            "mean": 0.72,
        },
    }
    plotting.save_lighting_panel(live, output_dir / "lighting")
    assert_png_and_pdf(output_dir / "lighting")

    with pytest.raises(ValueError):
        plotting.save_lighting_panel({"within_scan_jitter": []}, output_dir / "bad")


def test_save_pooling_and_confusion_panel(output_dir):
    rng = np.random.default_rng(0)
    plotting.save_pooling(
        rng.normal(0.8, 0.1, 200), rng.normal(0.55, 0.08, 40), output_dir / "pooling"
    )
    assert_png_and_pdf(output_dir / "pooling")

    # Map B is constant here — exactly the case that made the old scatter
    # version emit "Pearson r = nan".  It must render cleanly now.
    map_a = rng.random(40)
    map_b = np.ones(40)
    plotting.save_confusion_panel(
        map_a, map_b, output_dir / "confusion", labels=[f"id{index}" for index in range(40)]
    )
    assert_png_and_pdf(output_dir / "confusion")

    with pytest.raises(ValueError):
        plotting.save_confusion_panel(map_a, map_b[:10], output_dir / "bad")
