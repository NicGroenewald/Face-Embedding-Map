"""Tests for ``compare_embeddings.py`` using synthetic scans.

The script is a reporting tool, but two parts are load-bearing for trust:
the self-check that a vector has distance 0 from itself, and the exit code
when the data cannot be loaded.  Both are tested without touching the real
scan files or loading any model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import compare_embeddings as report
from fem.data import Scan, ScanSet


def _unit_scan(name: str, intensity: str, direction: str, seed: int) -> Scan:
    rng = np.random.default_rng(seed)
    frames = rng.normal(size=(10, 512)).astype(np.float32)
    frames /= np.linalg.norm(frames, axis=1, keepdims=True)
    return Scan(
        name=name,
        path=Path(f"{name}.npy"),
        intensity=intensity,
        direction=direction,
        frames=frames,
    )


def make_scan_set() -> ScanSet:
    """Four scans covering the pair types the report compares."""
    scans = [
        _unit_scan("room_front_1", "room", "front", 0),
        _unit_scan("room_front_2", "room", "front", 1),
        _unit_scan("flash_front_1", "flash", "front", 2),
        _unit_scan("flash_front_2", "flash", "front", 3),
    ]
    return ScanSet(scans=scans, subject="test", captured="2026-01-01")


def test_distance_stats_for_two_different_scans():
    scan_set = make_scan_set()
    frames_a = scan_set.by_name("room_front_1").frames
    frames_b = scan_set.by_name("flash_front_1").frames
    low, mean, high = report.distance_stats(frames_a, frames_b)

    assert low > 0.0  # different frames: no zero self-pair in the table
    assert 0.0 < mean <= high
    assert high < 2.0  # unit vectors


def test_symmetric_distance_stats_excludes_self_pairs():
    # Two frames: exactly one unordered distinct pair.  If the diagonal were
    # included the mean would be (0 + d + d + 0)/4 instead of d.
    first = np.zeros(512, dtype=np.float32)
    first[0] = 1.0
    second = np.zeros(512, dtype=np.float32)
    second[1] = 1.0
    frames = np.stack([first, second])

    low, mean, high = report.symmetric_distance_stats(frames)
    expected = np.sqrt(2.0)
    assert low == pytest.approx(expected, abs=1e-6)
    assert mean == pytest.approx(expected, abs=1e-6)
    assert high == pytest.approx(expected, abs=1e-6)

    with pytest.raises(ValueError):
        report.symmetric_distance_stats(first[None, :])


def test_self_check_passes_on_valid_scan_set():
    checks = report.self_check(make_scan_set())
    assert len(checks) == 3
    assert all(passed for _name, passed, _detail in checks)


def test_within_scan_and_same_condition_counts():
    scan_set = make_scan_set()
    jitter = report.within_scan_jitter(scan_set)
    assert [row["scan"] for row in jitter] == scan_set.names()
    # Distinct random frames: the minimum must be a real pair, never a
    # self-pair zero.  This is the regression test for the diagonal bug.
    assert all(row["min"] > 0.0 for row in jitter)

    same_condition = report.same_condition(scan_set)
    assert len(same_condition) == 2
    assert {row["condition"] for row in same_condition} == {"room/front", "flash/front"}


def test_comparison_pairs_only_includes_available_names():
    scan_set = make_scan_set()
    pairs = report.comparison_pairs(scan_set)
    assert ("room_front_1", "room_front_2") in pairs
    assert ("room_front_1", "flash_front_1") in pairs
    # Names the synthetic set does not have must not appear.
    assert all(a in scan_set.names() and b in scan_set.names() for a, b in pairs)


def test_main_returns_nonzero_for_missing_manifest(tmp_path, capsys):
    exit_code = report.main(["--manifest", str(tmp_path / "missing.json")])
    assert exit_code == 1
    assert "cannot load scans" in capsys.readouterr().err
