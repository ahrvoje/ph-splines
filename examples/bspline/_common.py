"""Shared, headless rendering helpers for the PH B-spline galleries."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ph_spline import PHBSpline

INK = "#172033"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
GRID = "#d9dee8"
SPLINE = "#1769aa"
BEFORE = "#8a94a6"
POINT = "#ed6a5a"
VECTOR = "#df8b18"

plt.rcParams.update(
    {
        "figure.facecolor": PAGE,
        "axes.facecolor": SURFACE,
    }
)


def normalized_samples(curve: PHBSpline, count: int = 600) -> np.ndarray:
    """Sample the internal normalized geometry without risking display overflow."""

    values = np.empty((count, 2), dtype=np.float64)
    for index, u in enumerate(np.linspace(0.0, 1.0, count)):
        span, local = curve._locate(float(u))
        value = curve._spans[span].point_normalized(local)
        values[index] = (value.real, value.imag)
    return values


def verify(curve: PHBSpline) -> None:
    """Exercise interpolation, regularity, and analytic distance inversion."""

    for index, knot in enumerate(curve._knots):
        if index >= curve.num_points:
            break
        assert np.array_equal(curve.point(float(knot)), curve.points[index])
    assert curve.diagnostics.min_regularity_ratio > 0.0
    for fraction in (0.0, 0.17, 0.5, 0.83, 1.0):
        target = fraction * curve.length
        recovered = float(curve.arc_length(curve.parameter_at_length(target)))
        tolerance = 1024.0 * np.finfo(float).eps * curve.length
        assert abs(recovered - target) <= tolerance + 8.0 * math.ulp(
            max(target, 1e-300)
        )


def render_curve(
    curve: PHBSpline,
    path: Path,
    title: str,
    note: str = "",
    *,
    before: np.ndarray | None = None,
    vectors: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    verify(curve)
    xy = normalized_samples(curve)
    points = np.column_stack(
        (curve._state.normalized_points.real, curve._state.normalized_points.imag)
    )
    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=105)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color=GRID, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    if before is not None:
        ax.plot(
            before[:, 0],
            before[:, 1],
            "--",
            color=BEFORE,
            linewidth=1.0,
            label="before edit",
        )
    ax.plot(
        points[:, 0], points[:, 1], "--", color=BEFORE, linewidth=0.8, label="nodes"
    )
    ax.plot(xy[:, 0], xy[:, 1], color=SPLINE, linewidth=2.0, label="PH B-spline")
    ax.scatter(
        points[:, 0], points[:, 1], s=17, facecolor="white", edgecolor=POINT, zorder=4
    )
    if vectors is not None:
        anchors, arrows = vectors
        norms = np.linalg.norm(arrows, axis=1)
        maximum = float(np.max(norms)) if norms.size else 0.0
        extent = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-12)
        display = arrows * (0.16 * extent / maximum) if maximum > 0.0 else arrows
        ax.quiver(
            anchors[:, 0],
            anchors[:, 1],
            display[:, 0],
            display[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=VECTOR,
            width=0.004,
        )
    subtitle = (
        f"{curve.num_points} nodes · {curve.num_spans} spans · "
        f"G{curve.verified_continuity.g_order} · degree {curve.degree}"
    )
    if note:
        subtitle += f"\n{note}"
    fig.suptitle(title.replace("_", " "), x=0.06, ha="left", color=INK, fontsize=11)
    ax.set_title(subtitle, loc="left", color=INK, fontsize=7.5)
    ax.legend(loc="best", frameon=False, fontsize=7.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)


def normalized_vectors(
    curve: PHBSpline,
    parameters: np.ndarray,
    evaluator,
) -> tuple[np.ndarray, np.ndarray]:
    anchors = np.empty((parameters.size, 2))
    arrows = np.empty_like(anchors)
    for index, u in enumerate(parameters):
        span, local = curve._locate(float(u))
        point = curve._spans[span].point_normalized(local)
        anchors[index] = (point.real, point.imag)
        arrows[index] = evaluator(float(u))
    return anchors, arrows
