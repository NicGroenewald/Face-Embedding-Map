"""Every figure in the project is drawn here, once.

Why one module
--------------
The comparison only means something if Map A and Map B are plotted with the
same colours, the same axis conventions and the same figure size.  If each
runner had its own ``plt.subplots`` block, any difference between the two
pictures could be a difference in settings rather than a difference between
the representations (the plan's trap #2).  So the rule is: runners compute
arrays, ``fem.plotting`` draws them.

Style is centralised
--------------------
A single ``_RC`` dict and a fixed colour-blind-safe palette (Paul Tol's) are
applied through ``plt.rc_context`` inside every saver.  No call site chooses
fonts, dpi, line widths or palette entries itself.

Every saver writes **both** a 200 dpi PNG and a vector PDF, at the same base
path: ``save_roc(curves, "figures/roc_curves")`` (or with ``.png`` / ``.pdf``)
produces ``figures/roc_curves.png`` and ``figures/roc_curves.pdf``.  Both are
needed: the repo README embeds PNGs, the paper includes the PDFs.

Public savers
-------------
Contract (DESIGN.md §4): ``save_scatter``, ``save_eigenfaces``, ``save_scree``,
``save_distance_histograms``, ``save_roc``, ``save_threshold_sweep``,
``save_heatmap``, ``save_metric_bars``, ``save_lighting_panel``.

Additional helpers used by the experiment runners so they do not reimplement
plotting: ``save_pooling`` (E3 pooling figure) and ``save_confusion_panel``
(E2 Map A vs Map B per-identity accuracy).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .config import IMAGES_PER_IDENTITY
from .maps import eigenfaces
from .metrics import eer, roc_auc, threshold_sweep

# ---------------------------------------------------------------------------
# Central style: one place to change every figure in the paper.
# ---------------------------------------------------------------------------
_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "figure.figsize": (3.4, 2.8),
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "lines.linewidth": 1.4,
    "lines.markersize": 3,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed TrueType outlines; searchable, journal-safe text
    "ps.fonttype": 42,
}

# Paul Tol's colour-blind-safe palette, plus consistent semantic colours.
MAP_A_COLOR = "#4477AA"
MAP_B_COLOR = "#EE6677"
GENUINE_COLOR = "#228833"
IMPOSTOR_COLOR = "#CC3311"
NEUTRAL = "#666666"
_SERIES_COLORS = ("#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#999999")

#: 40 identity colours, consistent between Map A and Map B scatter plots.
#:
#: 40 categories is more than any colour-blind-safe qualitative palette can
#: carry, so the priority is maximum separability with a deterministic
#: identity -> colour map.  Golden-ratio hue stepping spreads adjacent
#: identities far apart on the hue wheel; alternating saturation/value makes
#: the inevitably-close hues distinguishable.  Fixed constants, not a random
#: palette: Map A and Map B must colour identity *i* identically.
IDENTITY_COLORS: tuple = tuple(
    matplotlib.colors.hsv_to_rgb(
        (
            (index * 0.618033988749895 + 0.07) % 1.0,
            0.90 if index % 2 == 0 else 0.65,
            0.92 if (index // 2) % 2 == 0 else 0.68,
        )
    )
    for index in range(40)
)


def _save(fig, path: str | Path) -> None:
    """Write ``fig`` as 200 dpi PNG and vector PDF, then close it.

    This is the only place the project decides a figure's output format, so
    "png + pdf" cannot be forgotten on one figure out of twelve.
    """
    base = Path(path)
    if base.suffix.lower() in {".png", ".pdf"}:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=200)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def _series_colour(index: int, label: str) -> str:
    """Stable colour for a series: Map A/B keep their colours everywhere."""
    key = str(label).lower().replace("_", " ").replace("-", " ")
    if "map a" in key:
        return MAP_A_COLOR
    if "map b" in key:
        return MAP_B_COLOR
    return _SERIES_COLORS[index % len(_SERIES_COLORS)]


def _heat_colours(series: Sequence[str]) -> list:
    """Colours for a list of series names (Map A/Map B aware)."""
    return [_series_colour(index, name) for index, name in enumerate(series)]


def _identity_name(label, identity_names) -> str:
    """Resolve a display name for an identity label (list/mapping/None)."""
    if identity_names is None:
        return str(label)
    if isinstance(identity_names, Mapping):
        return str(identity_names.get(label, label))
    try:
        return str(identity_names[int(label)])
    except (IndexError, TypeError, ValueError):
        return str(label)


def _grid(ax) -> None:
    """Light horizontal grid; only where it helps read a value."""
    ax.grid(axis="y", color=NEUTRAL, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 2-D maps
# ---------------------------------------------------------------------------
def save_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
    title: str = "",
    identity_names=None,
    cmap=None,
) -> None:
    """Scatter a 2-D PCA projection, one colour per identity.

    ``coords`` is ``(N, 2)``; ``labels`` is ``(N,)`` (identity integers for
    Olivetti, or any hashable label).  Identity colour assignment is
    deterministic, so the Map A and Map B scatters colour identity *i* the
    same way.

    ``cmap`` optionally overrides the 40-colour identity palette (a colormap
    name or object), sampled evenly across the distinct labels.  ``identity_names``
    is used for a legend only when there are at most 12 identities; a 40-entry
    legend would be unreadable, so maps with 40 identities rely on colour only.
    """
    coords_array = np.asarray(coords, dtype=np.float64)
    labels_array = np.asarray(labels)
    if coords_array.ndim != 2 or coords_array.shape[1] != 2:
        raise ValueError(f"coords must be (N, 2), got {coords_array.shape}")
    if labels_array.shape[0] != coords_array.shape[0]:
        raise ValueError("coords and labels must have the same number of rows")

    unique_labels = np.unique(labels_array)
    if cmap is not None:
        colormap = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
        palette = [colormap(value) for value in np.linspace(0.0, 1.0, unique_labels.size)]
    else:
        palette = [IDENTITY_COLORS[i % len(IDENTITY_COLORS)] for i in range(unique_labels.size)]
    colour_of = {label: palette[i] for i, label in enumerate(unique_labels)}
    point_colours = [colour_of[label] for label in labels_array]

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        ax.scatter(
            coords_array[:, 0],
            coords_array[:, 1],
            c=point_colours,
            s=22,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.3,
        )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title(title or "2-D PCA projection")
        ax.set_aspect("equal", adjustable="datalim")
        if identity_names is not None and unique_labels.size <= 12:
            handles = [
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    color=colour_of[label],
                    label=_identity_name(label, identity_names),
                )
                for label in unique_labels
            ]
            ax.legend(handles=handles, loc="best", frameon=False, fontsize=6, ncol=2)
        _save(fig, path)


def save_eigenfaces(
    components: np.ndarray,
    path: str | Path,
    n: int = 16,
    image_shape: tuple[int, int] = (64, 64),
) -> None:
    """Render the first ``n`` principal components as a grid of faces.

    Each face is scaled to its own symmetric range: the components' relative
    amplitudes are already shown by the scree plot, and scaling each one makes
    the *shape* of the later components readable instead of near-black.
    """
    faces = eigenfaces(components, n=n, image_shape=image_shape)
    count = faces.shape[0]
    columns = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            rows, columns, figsize=(columns * 1.15, rows * 1.15), squeeze=False
        )
        flat_axes = axes.ravel()
        for index, ax in enumerate(flat_axes):
            if index >= count:
                ax.axis("off")
                continue
            face = faces[index]
            limit = float(np.abs(face).max()) or 1.0
            ax.imshow(face, cmap="gray", vmin=-limit, vmax=limit, interpolation="nearest")
            ax.set_title(f"PC {index + 1}", fontsize=6, pad=2)
            ax.axis("off")
        fig.suptitle("Eigenfaces: principal components of the raw pixels", fontsize=8)
        _save(fig, path)


# ---------------------------------------------------------------------------
# variance
# ---------------------------------------------------------------------------
def save_scree(evr, path: str | Path, title: str = "") -> None:
    """Per-component and cumulative explained variance.

    ``evr`` may be one array (one map) or a mapping ``{label: evr}`` so the
    paper's two-map scree figure is a single call with consistent colours.
    """
    if isinstance(evr, Mapping):
        series = {str(label): np.asarray(values, dtype=np.float64).ravel() for label, values in evr.items()}
    else:
        series = {"explained variance": np.asarray(evr, dtype=np.float64).ravel()}
    if not series:
        raise ValueError("evr is empty")
    for label, ratios in series.items():
        if ratios.size == 0:
            raise ValueError(f"evr series {label!r} is empty")

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
        for index, (label, ratios) in enumerate(series.items()):
            colour = _series_colour(index, label)
            components = np.arange(1, ratios.size + 1)
            axes[0].plot(components, ratios, color=colour, label=label, marker="o", markersize=2)
            axes[1].plot(components, np.cumsum(ratios), color=colour, label=label)
        for milestone in (25, 50, 75, 90, 95):
            axes[1].axhline(milestone / 100.0, color=NEUTRAL, linewidth=0.5, linestyle=":")
            axes[1].text(
                0.01,
                milestone / 100.0 + 0.01,
                f"{milestone}%",
                transform=axes[1].get_yaxis_transform(),
                fontsize=6,
                color=NEUTRAL,
            )
        axes[0].set_xlabel("principal component")
        axes[0].set_ylabel("explained variance ratio")
        axes[0].set_title("Scree (per component)")
        axes[1].set_xlabel("principal component")
        axes[1].set_ylabel("cumulative explained variance")
        axes[1].set_ylim(0.0, 1.02)
        axes[1].set_title("Cumulative variance")
        if len(series) > 1:
            axes[1].legend(loc="lower right", frameon=False)
        fig.suptitle(title or "Explained variance", fontsize=9)
        _save(fig, path)


# ---------------------------------------------------------------------------
# genuine / impostor verification
# ---------------------------------------------------------------------------
def save_distance_histograms(
    data: Mapping[str, tuple[np.ndarray, np.ndarray]],
    path: str | Path,
    xlabel: str = "Euclidean distance",
) -> None:
    """Genuine vs impostor distance distributions.

    ``data`` is ``{label: (genuine_distances, impostor_distances)}`` — one
    panel per label (Map A, Map B).  Genuine and impostor keep the same two
    colours in every panel, so panels are directly comparable.
    """
    if not isinstance(data, Mapping) or not data:
        raise ValueError("data must be a non-empty {label: (genuine, impostor)} mapping")

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(data), figsize=(3.2 * len(data), 2.7), squeeze=False)
        for ax, (label, pair) in zip(axes.ravel(), data.items()):
            genuine = np.asarray(pair[0], dtype=np.float64).ravel()
            impostor = np.asarray(pair[1], dtype=np.float64).ravel()
            if genuine.size == 0 or impostor.size == 0:
                raise ValueError(f"series {label!r} has an empty distance array")
            bins = np.histogram_bin_edges(np.concatenate([genuine, impostor]), bins=60)
            ax.hist(genuine, bins=bins, density=True, color=GENUINE_COLOR, alpha=0.45, label="genuine")
            ax.hist(impostor, bins=bins, density=True, color=IMPOSTOR_COLOR, alpha=0.35, label="impostor")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("density")
            ax.set_title(str(label))
            auc = roc_auc(genuine, impostor)
            equal_error, _ = eer(genuine, impostor)
            ax.text(
                0.02,
                0.97,
                f"AUC {auc:.3f}\nEER {equal_error:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )
            ax.legend(loc="upper right", frameon=False)
        fig.suptitle("Genuine vs impostor distances", fontsize=9)
        _save(fig, path)


def save_roc(curves: Mapping[str, tuple[np.ndarray, np.ndarray]], path: str | Path) -> None:
    """ROC curves with EER points marked.

    ``curves`` is ``{label: (genuine_distances, impostor_distances)}``; the
    curve is the empirical ROC of the distance score (accept iff distance is
    below a threshold).
    """
    if not isinstance(curves, Mapping) or not curves:
        raise ValueError("curves must be a non-empty {label: (genuine, impostor)} mapping")

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(3.5, 3.2))
        handles = []
        for index, (label, (genuine, impostor)) in enumerate(curves.items()):
            colour = _series_colour(index, label)
            sweep = threshold_sweep(genuine, impostor)
            true_positive = 1.0 - sweep["frr"]
            false_positive = sweep["far"]
            auc = roc_auc(genuine, impostor)
            equal_error, _ = eer(genuine, impostor)
            ax.plot(false_positive, true_positive, color=colour, label=f"{label} (AUC {auc:.3f})")
            ax.plot([equal_error], [1.0 - equal_error], marker="o", linestyle="none", color=colour)
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    color=colour,
                    label=f"{label} EER {equal_error:.3f}",
                )
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color=NEUTRAL, linewidth=0.8, label="chance")
        ax.set_xlabel("false acceptance rate (FAR)")
        ax.set_ylabel("true acceptance rate (1 - FRR)")
        ax.set_xlim(-0.01, 1.0)
        ax.set_ylim(0.0, 1.01)
        ax.set_title("ROC: genuine vs impostor trials")
        ax.legend(handles=handles + [Line2D([], [], linestyle="--", color=NEUTRAL, label="chance")],
                  loc="lower right", frameon=False, fontsize=6)
        _save(fig, path)


def save_threshold_sweep(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray]],
    path: str | Path,
    separate_panels: bool = False,
) -> None:
    """FAR and FRR as functions of the decision threshold, one colour per map.

    Solid = FAR, dashed = FRR, dotted vertical = the EER threshold.  This is
    the figure for the security/convenience dial: pick an x, read both errors.

    ``separate_panels=True`` draws one panel per map instead of overlaying
    them.  That is required when the maps' distance scales differ by an order
    of magnitude (raw-pixel distances ~10x embedding distances): on a shared
    x-axis one map's curves collapse into a vertical line at zero and the
    other's fill the figure, which hides the very thing the figure is for.
    """
    if not isinstance(curves, Mapping) or not curves:
        raise ValueError("curves must be a non-empty {label: (genuine, impostor)} mapping")

    with plt.rc_context(_RC):
        if separate_panels:
            fig, axes = plt.subplots(1, len(curves), figsize=(3.1 * len(curves), 2.9), squeeze=False)
            panel_axes = axes.ravel()
        else:
            fig, ax = plt.subplots(figsize=(4.2, 2.9))
            panel_axes = [ax]
        for index, (label, (genuine, impostor)) in enumerate(curves.items()):
            colour = _series_colour(index, label)
            sweep = threshold_sweep(genuine, impostor)
            current = panel_axes[0] if not separate_panels else panel_axes[index]
            current.plot(sweep["thresholds"], sweep["far"], color=colour, label=f"{label} FAR")
            current.plot(
                sweep["thresholds"], sweep["frr"], color=colour, linestyle="--", label=f"{label} FRR"
            )
            equal_error, threshold = eer(genuine, impostor)
            current.axvline(threshold, color=colour, linestyle=":", linewidth=0.8)
            current.plot([threshold], [equal_error], marker="o", linestyle="none", color=colour)
            if separate_panels:
                current.set_title(str(label))
                current.legend(loc="best", frameon=False, fontsize=6)
                current.set_xlabel("decision threshold")
                current.set_ylabel("error rate")
                current.set_ylim(0.0, 1.02)
            else:
                current.set_xlabel("decision threshold (accept if distance ≤ threshold)")
                current.set_ylabel("error rate")
                current.set_ylim(0.0, 1.02)
        if not separate_panels:
            ax.set_title("FAR and FRR vs threshold")
            ax.legend(loc="best", frameon=False, fontsize=6, ncol=2)
        _save(fig, path)


# ---------------------------------------------------------------------------
# matrices and comparisons
# ---------------------------------------------------------------------------
def save_heatmap(
    matrix: np.ndarray,
    tick_labels: Sequence[str],
    path: str | Path,
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Square distance matrix with labelled axes.

    Cell values are annotated for matrices up to 14x14 (the live study is
    12x12).  The colour scale is shared across cells; annotating makes the
    exact distances readable without judging colour by eye.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {array.shape}")
    tick_labels = [str(label) for label in tick_labels]
    if len(tick_labels) != array.shape[0] or len(tick_labels) != array.shape[1]:
        raise ValueError(
            f"{len(tick_labels)} tick labels for a {array.shape[0]}x{array.shape[1]} matrix"
        )

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        image = ax.imshow(
            array,
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        if array.shape[0] <= 14:
            low, high = image.norm.vmin, image.norm.vmax
            midpoint = 0.5 * (low + high)
            for row in range(array.shape[0]):
                for column in range(array.shape[1]):
                    value = array[row, column]
                    ax.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="black" if value > midpoint else "white",
                    )
        ax.set_xticks(range(array.shape[1]))
        ax.set_yticks(range(array.shape[0]))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(tick_labels, fontsize=6)
        ax.set_title(title or "Mean Euclidean distance between scans")
        colourbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colourbar.set_label("Euclidean distance")
        _save(fig, path)


def save_metric_bars(metrics, path: str | Path, ylabel: str = "") -> None:
    """Side-by-side bars for the headline metrics, one panel per metric.

    Two accepted shapes for ``metrics`` (the runners pass whichever is natural):

    * ``{metric_name: (map_a_value, map_b_value), ...}`` — groups are metrics,
      series are Map A / Map B;
    * ``{"Map A": {metric_name: value, ...}, "Map B": {...}}`` — series names
      are the outer keys.

    One panel per metric rather than one shared axis, because silhouette,
    accuracy and EER live on different scales and a single axis would make
    the comparison unreadable.
    """
    series_names, metric_names, values = _normalise_metric_bars(metrics)
    colours = _heat_colours(series_names)

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            1, len(metric_names), figsize=(max(3.6, 1.7 * len(metric_names) + 1.0), 2.7), squeeze=False
        )
        flat_axes = axes.ravel()
        for metric_index, (ax, metric) in enumerate(zip(flat_axes, metric_names)):
            bar_values = [values[series][metric] for series in series_names]
            positions = np.arange(len(series_names))
            bars = ax.bar(positions, bar_values, color=colours, width=0.7)
            for bar, value in zip(bars, bar_values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )
            ax.set_xticks(positions)
            ax.set_xticklabels(series_names, rotation=0, fontsize=6)
            ax.set_title(str(metric).replace("_", " "))
            _grid(ax)
            low = min(0.0, min(bar_values) * 1.1)
            high = max(bar_values) * 1.18 if max(bar_values) > 0 else 1.0
            ax.set_ylim(low, high if high > low else low + 1.0)
            if metric_index == 0:
                ax.set_ylabel(ylabel or "score")
            else:
                ax.set_ylabel("")
        fig.suptitle("Map A vs Map B: headline metrics", fontsize=9)
        _save(fig, path)


def _normalise_metric_bars(metrics) -> tuple[list[str], list[str], dict[str, dict[str, float]]]:
    """Turn either accepted ``metrics`` shape into (series, metrics, values)."""
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("metrics must be a non-empty mapping")
    outer_values = list(metrics.values())

    if all(isinstance(value, Mapping) for value in outer_values):
        series_names = [str(name) for name in metrics.keys()]
        metric_names = [str(name) for name in outer_values[0].keys()]
        if not metric_names:
            raise ValueError("nested metrics are empty")
        values: dict[str, dict[str, float]] = {}
        for series, series_values in zip(series_names, outer_values):
            if [str(name) for name in series_values.keys()] != metric_names:
                raise ValueError("every series must contain the same metric names")
            values[series] = {str(name): float(series_values[name]) for name in series_values}
        return series_names, metric_names, values

    pairs = [np.asarray(value, dtype=np.float64).ravel() for value in outer_values]
    if not all(pair.size == 2 for pair in pairs):
        raise ValueError(
            "metrics must be {metric: (map_a, map_b)} or {series: {metric: value}}"
        )
    series_names = ["Map A", "Map B"]
    metric_names = [str(name) for name in metrics.keys()]
    values = {
        series_names[index]: {
            metric: float(pair[index]) for metric, pair in zip(metric_names, pairs)
        }
        for index in range(2)
    }
    return series_names, metric_names, values


# ---------------------------------------------------------------------------
# lighting study panels (Map B live scans)
# ---------------------------------------------------------------------------
def save_lighting_panel(live: Mapping, path: str | Path) -> None:
    """Noise floor vs intensity effect vs direction effect, with min-max whiskers.

    ``live`` is the ``results/live.json`` payload: the four categories are
    ``within_scan_jitter``, ``same_condition``, ``intensity_effect`` and
    ``direction_effect``.  The first three are lists of per-scan/per-pair
    ``{"min","mean","max"}`` dicts; the intensity effect is one such dict.
    Means are compared as bars, observed min-max ranges as whiskers, so the
    reader sees both "how big is the effect" and "how noisy is the estimate".
    """
    if not isinstance(live, Mapping):
        raise ValueError("live must be the results/live.json mapping")

    categories = (
        ("within_scan_jitter", "frame jitter\n(within scan)", "#4477AA"),
        ("same_condition", "same condition\n(repeat scans)", "#66CCEE"),
        ("intensity_effect", "intensity\n(room vs flash)", GENUINE_COLOR),
        ("direction_effect", "direction\n(flash only)", IMPOSTOR_COLOR),
    )
    present: list[tuple[str, float, float, float, str]] = []
    for key, label, colour in categories:
        if key not in live:
            continue
        mean, low, high = _range_stats(live[key])
        present.append((label, mean, low, high, colour))
    if len(present) < 2:
        raise ValueError("live payload has fewer than two lighting categories")

    labels = [item[0] for item in present]
    means = np.array([item[1] for item in present])
    lower = means - np.array([item[2] for item in present])
    upper = np.array([item[3] for item in present]) - means
    colours = [item[4] for item in present]
    positions = np.arange(len(present))

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.6, 2.9))
        bars = ax.bar(positions, means, color=colours, width=0.65)
        ax.errorbar(
            positions,
            means,
            yerr=[lower, upper],
            fmt="none",
            ecolor=NEUTRAL,
            capsize=3,
            linewidth=0.8,
        )
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylabel("mean Euclidean distance")
        ax.set_title("Frame jitter vs lighting effect (min–max whiskers)")
        _grid(ax)
        _save(fig, path)


def _range_stats(entry) -> tuple[float, float, float]:
    """(mean, min, max) for one lighting statistic.

    Accepts the three shapes ``results/live.json`` uses: a single
    ``{"min","mean","max"}`` dict (intensity effect), a list of such dicts
    (within-scan jitter, same-condition pairs), or the direction-effect dict
    whose per-pair statistics live under ``"pairs"``.
    """
    if isinstance(entry, Mapping):
        pairs = entry.get("pairs")
        if isinstance(pairs, (list, tuple)) and pairs:
            return _range_stats(list(pairs))
        if all(key in entry for key in ("mean", "min", "max")):
            return float(entry["mean"]), float(entry["min"]), float(entry["max"])
        raise ValueError(f"lighting statistic dict needs mean/min/max, got keys {list(entry)}")
    if isinstance(entry, (list, tuple)) and entry:
        if not all(isinstance(item, Mapping) for item in entry):
            raise ValueError("a lighting statistic list must contain {min,mean,max} dicts")
        means = [float(item["mean"]) for item in entry]
        lows = [float(item["min"]) for item in entry]
        highs = [float(item["max"]) for item in entry]
        return float(np.mean(means)), min(lows), max(highs)
    raise ValueError(f"cannot read lighting statistics from {type(entry).__name__}")


def save_pooling(
    frame_distances: np.ndarray,
    pooled_distances: np.ndarray,
    path: str | Path,
    title: str = "",
) -> None:
    """Frame-level vs scan-mean distance distributions (E3 pooling figure).

    Caller passes the *same* scan pairs measured two ways — all frame pairs vs
    the two pooled scan means — so the leftward shift is the averaging effect,
    not a different comparison set.
    """
    frame = np.asarray(frame_distances, dtype=np.float64).ravel()
    pooled = np.asarray(pooled_distances, dtype=np.float64).ravel()
    if frame.size == 0 or pooled.size == 0:
        raise ValueError("both distance arrays must be non-empty")

    reduction = (1.0 - pooled.mean() / frame.mean()) * 100.0 if frame.mean() > 0 else 0.0
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.3, 2.9))
        bins = np.histogram_bin_edges(np.concatenate([frame, pooled]), bins=60)
        ax.hist(
            frame,
            bins=bins,
            density=True,
            color=NEUTRAL,
            alpha=0.45,
            label=f"frame-level (mean {frame.mean():.3f})",
        )
        ax.hist(
            pooled,
            bins=bins,
            density=True,
            color=MAP_A_COLOR,
            alpha=0.55,
            label=f"scan-mean (mean {pooled.mean():.3f})",
        )
        ax.set_xlabel("Euclidean distance")
        ax.set_ylabel("density")
        ax.set_title(title or "Pooling frames at enrolment reduces distance")
        ax.text(
            0.02,
            0.97,
            f"{reduction:.1f}% closer after averaging",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
        )
        ax.legend(loc="upper right", frameon=False, fontsize=6)
        _save(fig, path)


def save_confusion_panel(
    per_identity_a: np.ndarray,
    per_identity_b: np.ndarray,
    path: str | Path,
    labels: Sequence[str] | None = None,
) -> None:
    """Per-identity 1-NN accuracy, Map A as sorted bars against Map B.

    Why bars, not a Map A-vs-Map B scatter
    --------------------------------------
    An ``(a, b)`` scatter is only informative when both axes vary.  If one map
    identifies every image correctly, all its points collapse onto a single
    horizontal line, the markers overlap and the Pearson correlation is
    undefined — the figure says nothing.  Bars sorted by Map A accuracy plus a
    reference line for Map B stay readable in that (actual) case: you can see
    *which* identities Map A gets wrong and how far below Map B they are.

    ``per_identity_a``/``per_identity_b`` are arrays of per-identity accuracy
    (fraction of that identity's images classified correctly, one value per
    identity in identity order).  ``labels`` optionally supplies identity
    names for the axis ticks.  Counts are derived from
    ``IMAGES_PER_IDENTITY`` so the caption-friendly annotation is generic, not
    hardcoded to 400 images.
    """
    map_a = np.asarray(per_identity_a, dtype=np.float64).ravel()
    map_b = np.asarray(per_identity_b, dtype=np.float64).ravel()
    if map_a.shape != map_b.shape:
        raise ValueError("per-identity accuracy arrays must have the same length")
    if map_a.size < 2:
        raise ValueError("need at least two identities")
    if labels is not None and len(labels) != map_a.size:
        raise ValueError(
            f"received {len(labels)} identity labels for {map_a.size} identities"
        )

    order = np.argsort(-map_a, kind="stable")  # best Map A identity first
    sorted_a = map_a[order]
    identity_names = (
        [str(labels[index]) for index in order]
        if labels is not None
        else [f"identity {index}" for index in order]
    )
    positions = np.arange(map_a.size)
    map_b_mean = float(map_b.mean())

    total_images = int(map_a.size * IMAGES_PER_IDENTITY)
    correct_a = int(round(float(map_a.sum() * IMAGES_PER_IDENTITY)))
    correct_b = int(round(float(map_b.sum() * IMAGES_PER_IDENTITY)))
    counts_text = (
        f"Map A: {correct_a}/{total_images} images correct "
        f"({total_images - correct_a} errors)    "
        f"Map B: {correct_b}/{total_images} images correct "
        f"({total_images - correct_b} errors)"
    )

    # With 40 identities, label every identity only when they fit; otherwise
    # show ~20 evenly spaced labels and rely on sorting for the rest.
    step = 1 if map_a.size <= 20 else int(np.ceil(map_a.size / 20))
    tick_positions = positions[::step]
    tick_labels = [identity_names[index] for index in tick_positions]

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.0, max(3.2, 0.13 * map_a.size + 1.5)))
        ax.barh(positions, sorted_a, height=0.8, color=MAP_A_COLOR, alpha=0.9)
        reference = ax.axvline(
            map_b_mean,
            color=MAP_B_COLOR,
            linestyle="--",
            linewidth=1.1,
            label=f"Map B mean {map_b_mean:.3f}",
        )
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=6)
        ax.set_ylim(map_a.size - 0.5, -0.5)  # best Map A identity at the top
        ax.set_xlim(0.0, 1.02)
        ax.set_xlabel(counts_text, fontsize=7)
        ax.set_ylabel("identity (sorted by Map A accuracy)")
        ax.set_title("Per-identity 1-NN LOO accuracy")
        _grid(ax)
        ax.legend(handles=[reference], loc="lower right", frameon=False, fontsize=7)
        _save(fig, path)
