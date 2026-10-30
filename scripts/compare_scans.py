"""Quick scan-distance report for the live embeddings (human-readable table).

What this is
------------
A small CLI that prints the within-scan jitter, the same-condition repeat
distances, and the lighting-direction/intensity comparisons for the 12 saved
scans.  It is the sanity-check tool: the full analysis with JSON outputs lives
in ``analysis/run_live.py`` and is the source of the paper numbers.

Why it reads the manifest
-------------------------
The old version of this script hardcoded a filename -> condition dictionary.
That duplicated metadata already stored in ``data/scans.json`` and silently
broke when a file was renamed.  Condition metadata now comes from
``fem.data.load_scans()``, and distances are vectorised through
``fem.metrics.pairwise_distances`` instead of nested Python loops.

Distances are Euclidean between unit vectors; because both operand sets are
unit norm the cosine similarity is ``1 - d**2 / 2`` and is printed alongside so
the numbers can be read on the conventional 0..1 cosine scale too.

Usage
-----
    export MPLCONFIGDIR="$PWD/.cache/mpl"
    python scripts/compare_scans.py   # from an environment with requirements.txt installed

Exit status is 1 if the data cannot be loaded or the self-check fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/compare_scans.py` from the repo root (the script's own
# directory is on sys.path, the repo root is not). Importing it as
# `scripts.compare_scans` is unaffected.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fem.config import SCANS_MANIFEST
from fem.data import ScanSet, load_scans
from fem.metrics import cosine_from_euclidean, pairwise_distances


def distance_stats(A: np.ndarray, B: np.ndarray) -> tuple[float, float, float]:
    """Distances between two *different* frame sets as ``(min, mean, max)``.

    Every ordered pair (row of ``A``, row of ``B``) is included, so two
    10-frame scans contribute 100 values.  Self-pairs are irrelevant here
    because the two operand sets are different scans.

    Do not call this with ``A is B`` / equal arrays to measure within-set
    jitter: the 10 self-pairs would contribute zeros and dilute the mean.  Use
    ``symmetric_distance_stats`` for that.
    """
    distances = pairwise_distances(A, B)
    return float(distances.min()), float(distances.mean()), float(distances.max())


def symmetric_distance_stats(X: np.ndarray) -> tuple[float, float, float]:
    """Unordered *distinct* pairs within one frame set: ``(min, mean, max)``.

    Uses the strict upper triangle of the distance matrix, so a 10-frame scan
    contributes the 45 frame pairs the "within-scan jitter" protocol defines —
    no self-pairs, no double-counted ordered pairs.  This is what makes the
    jitter comparable to the 100 pairs of a two-scan comparison.
    """
    distances = pairwise_distances(X)
    if distances.shape[0] < 2:
        raise ValueError("need at least two frames to measure within-set distances")
    upper = np.triu_indices_from(distances, k=1)
    values = distances[upper]
    return float(values.min()), float(values.mean()), float(values.max())


def within_scan_jitter(scan_set: ScanSet) -> list[dict]:
    """Noise floor per scan: stats over all 45 frame pairs inside the scan."""
    rows = []
    for scan in scan_set:
        low, mean, high = symmetric_distance_stats(scan.frames)
        rows.append({"scan": scan.name, "min": low, "mean": mean, "max": high})
    return rows


def same_condition(scan_set: ScanSet) -> list[dict]:
    """Between-repeat distances for each lighting condition (100 frame pairs)."""
    rows = []
    for condition, scans in scan_set.by_condition().items():
        if len(scans) != 2:
            continue
        low, mean, high = distance_stats(scans[0].frames, scans[1].frames)
        rows.append({"condition": condition, "min": low, "mean": mean, "max": high})
    return rows


def comparison_pairs(scan_set: ScanSet) -> list[tuple[str, str]]:
    """Pairs worth printing: repeat baselines first, then the lighting effects."""
    names = set(scan_set.names())

    def present(pair: tuple[str, str]) -> bool:
        return pair[0] in names and pair[1] in names

    candidates = [
        # baselines: same condition, same person
        ("room_front_1", "room_front_2"),
        ("room_left_1", "room_left_2"),
        ("flash_front_1", "flash_front_2"),
        ("flash_left_1", "flash_left_2"),
        ("flash_right_1", "flash_right_2"),
        ("flash_bottom_1", "flash_bottom_2"),
        # intensity only (front light)
        ("room_front_1", "flash_front_1"),
        # direction only (flash light)
        ("flash_front_1", "flash_left_1"),
        ("flash_front_1", "flash_right_1"),
        ("flash_front_1", "flash_bottom_1"),
        ("flash_left_1", "flash_right_1"),
        ("flash_left_1", "flash_bottom_1"),
        ("flash_right_1", "flash_bottom_1"),
        # room vs a different direction (confounded; shown last)
        ("room_front_1", "flash_left_1"),
        ("room_front_1", "flash_bottom_1"),
        ("room_front_1", "room_left_1"),
    ]
    return [pair for pair in candidates if present(pair)]


def self_check(scan_set: ScanSet) -> list[tuple[str, bool, str]]:
    """Real checks instead of assertions on floats.

    The critical invariant is that a vector is distance 0 from itself (so the
    scale has an absolute zero) and that the Euclidean <-> cosine conversion is
    consistent with it.  These are reported as PASS/FAIL rows; a failure makes
    the script exit non-zero rather than raise somewhere in the table.
    """
    frame = scan_set.frames()[0]
    distance_to_self = float(pairwise_distances(frame[None, :], frame[None, :])[0, 0])
    cosine_to_self = float(cosine_from_euclidean(distance_to_self))
    norms = np.linalg.norm(scan_set.frames(), axis=1)
    # Distances are computed in float32 for speed; an identical pair can come
    # back as ~1e-8 rather than exactly 0.  A tolerance is the honest check —
    # the point is that the scale has an absolute zero, not bit-exact maths.
    return [
        (
            "single face -> distance 0",
            distance_to_self <= 1e-6,
            f"distance {distance_to_self:.1e}",
        ),
        (
            "cosine(0) == 1",
            abs(cosine_to_self - 1.0) <= 1e-12,
            f"cosine {cosine_to_self:.6f}",
        ),
        (
            "all rows unit norm",
            bool(np.allclose(norms, 1.0, atol=1e-5)),
            f"norm range [{norms.min():.6f}, {norms.max():.6f}]",
        ),
    ]


def _format_cosine(mean_distance: float) -> str:
    return f"{cosine_from_euclidean(mean_distance):.3f}"


def print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    """Fixed-width table so successive runs are diffable by eye."""
    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(len(header))]
    line = "  ".join(value.ljust(widths[i]) for i, value in enumerate(header))
    print(f"\n{title}")
    print(line)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def build_report(scan_set: ScanSet) -> int:
    """Print the full report; return a process exit code (0 okay, 1 self-check failed)."""
    print(f"Loaded {len(scan_set)} scans x {scan_set.frames_per_scan} frames "
          f"({scan_set.subject}, captured {scan_set.captured})")
    print(f"Embedding dimension: {scan_set.frames().shape[1]}")

    print_table(
        "Scans",
        ["scan", "shape", "condition"],
        [
            [scan.name, f"{scan.frames.shape}", scan.condition]
            for scan in scan_set
        ],
    )

    checks = self_check(scan_set)
    print_table(
        "Self-check",
        ["check", "result", "detail"],
        [[name, "PASS" if passed else "FAIL", detail] for name, passed, detail in checks],
    )
    checks_ok = all(passed for _name, passed, _detail in checks)

    jitter = within_scan_jitter(scan_set)
    print_table(
        "Within-scan jitter (45 frame pairs per scan)",
        ["scan", "min", "mean", "max", "cos@mean"],
        [
            [row["scan"], f"{row['min']:.3f}", f"{row['mean']:.3f}", f"{row['max']:.3f}",
             _format_cosine(row["mean"])]
            for row in jitter
        ],
    )
    print(f"\nmean within-scan jitter: {np.mean([row['mean'] for row in jitter]):.3f}")

    print_table(
        "Same condition, repeated scan (100 frame pairs)",
        ["condition", "min", "mean", "max", "cos@mean"],
        [
            [row["condition"], f"{row['min']:.3f}", f"{row['mean']:.3f}", f"{row['max']:.3f}",
             _format_cosine(row["mean"])]
            for row in same_condition(scan_set)
        ],
    )

    print_table(
        "Selected scan-pair distances",
        ["A", "B", "min", "mean", "max", "cos@mean"],
        [
            [a, b, f"{low:.3f}", f"{mean:.3f}", f"{high:.3f}", _format_cosine(mean)]
            for a, b in comparison_pairs(scan_set)
            for low, mean, high in [distance_stats(scan_set.by_name(a).frames, scan_set.by_name(b).frames)]
        ],
    )
    return 0 if checks_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print distance tables for the saved live face scans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help=f"scan manifest (default: {SCANS_MANIFEST})",
    )
    args = parser.parse_args(argv)

    try:
        scan_set = load_scans(args.manifest)
    except (FileNotFoundError, ValueError) as error:
        print(f"[compare_scans] cannot load scans: {error}", file=sys.stderr)
        return 1

    return build_report(scan_set)


if __name__ == "__main__":
    sys.exit(main())
