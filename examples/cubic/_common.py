"""Shared plotting and verification helpers for the example galleries.

Colors follow the validated reference palette of the data-viz method
(light mode): series-1 blue for the spline, series-2 orange for the
curvature comb, muted grays for chrome, ink tokens for text.
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    ),
)

from ph_spline import CubicPHSplineOpen

# Reference palette (light mode), see dataviz references/palette.md.
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_SPLINE = "#2a78d6"  # slot 1 blue
SERIES_COMB = "#eb6834"  # slot 2 orange

plt.rcParams.update(
    {
        "figure.facecolor": PAGE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": MUTED,
        "axes.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "font.family": "sans-serif",
        "font.size": 9,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
    }
)


def verify_contract(curve: CubicPHSplineOpen) -> None:
    """Independent spot-check of the post-construction invariants."""
    eps = np.finfo(float).eps
    for left, right in zip(curve._segments[:-1], curve._segments[1:]):
        tl = left.tangent_local(1.0)
        tr = right.tangent_local(0.0)
        assert math.hypot(tl[0] - tr[0], tl[1] - tr[1]) < 1e-12
        kl, kr = left.curvature_local(1.0), right.curvature_local(0.0)
        assert abs(kl - kr) / max(abs(kl), abs(kr), eps) < 1e-10
    L = curve.arc_length(1.0)
    assert math.isfinite(L) and L > 0.0
    for f in (0.25, 0.75):
        s = f * L
        u = curve.parameter_at_length(s)
        assert abs(curve.arc_length(u) - s) <= 256.0 * eps * L + 8.0 * math.ulp(s)


def spline_stats(curve: CubicPHSplineOpen) -> str:
    m = curve._m
    kind = "straight" if curve._is_straight else ("ccw" if curve._tau > 0 else "cw")
    L = curve.arc_length(1.0)
    kappas = [abs(curve.signed_curvature(float(u))) for u in np.linspace(0, 1, 121)]
    k_lo, k_hi = min(kappas), max(kappas)
    if k_hi == 0.0:
        k_txt = "kappa = 0"
    elif k_hi / max(k_lo, 1e-300) > 1e3 or k_hi >= 1e4 or k_hi < 1e-3:
        k_txt = f"|kappa| in [{k_lo:.2e}, {k_hi:.2e}]"
    else:
        k_txt = f"|kappa| in [{k_lo:.3g}, {k_hi:.3g}]"
    L_txt = f"L = {L:.4g}" if 1e-3 <= L < 1e6 else f"L = {L:.3e}"
    return f"{m + 1} pts / {m} segments, {kind} · {L_txt} · {k_txt}"


def _display_transform(points: np.ndarray) -> tuple[float, float, float, str]:
    """Shift/scale for display when coordinates are extreme; note if used."""
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    span = float(np.max(points.max(axis=0) - points.min(axis=0)))
    if span == 0.0:
        span = 1.0
    extreme = span < 1e-3 or span > 1e4 or float(np.max(np.abs(center))) > 50.0 * span
    if not extreme:
        return 0.0, 0.0, 1.0, ""
    note = (
        f"display recentered at ({center[0]:.3g}, {center[1]:.3g}), unit = {span:.3g}"
    )
    return float(center[0]), float(center[1]), span, note


def plot_example(
    index: int,
    name: str,
    points: list,
    out_dir: str,
    note: str = "",
    curve: CubicPHSplineOpen | None = None,
    comb: bool = True,
) -> CubicPHSplineOpen:
    """Construct (if needed), verify, and render one example to PNG."""
    if curve is None:
        curve = CubicPHSplineOpen(points)
    verify_contract(curve)

    pts = np.asarray(curve._points, dtype=np.float64)
    m = curve._m
    cx, cy, sc, tr_note = _display_transform(pts)

    def disp(arr: np.ndarray) -> np.ndarray:
        return (arr - np.array([cx, cy])) / sc

    n_samples = min(max(400, 14 * m), 4000)
    us = np.linspace(0.0, 1.0, n_samples)
    curve_xy = disp(np.array([curve.point(float(u)) for u in us]))
    pts_d = disp(pts)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=110)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, which="major")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = []
    # Input polyline (reference geometry, recessive).
    if m <= 400:
        (h_poly,) = ax.plot(
            pts_d[:, 0],
            pts_d[:, 1],
            linestyle=(0, (4, 3)),
            linewidth=0.9 if m <= 120 else 0.6,
            color=BASELINE,
            zorder=2,
            label="input polyline",
        )
        handles.append(h_poly)

    # Curvature comb (orange), drawn outward from the curve.
    if comb and not curve._is_straight:
        uc = np.linspace(0.0, 1.0, min(max(120, 6 * m), 900))
        pc = disp(np.array([curve.point(float(u)) for u in uc]))
        K = np.array([curve.curvature_vector(float(u)) for u in uc])
        k_abs = np.hypot(K[:, 0], K[:, 1])
        k_max = float(k_abs.max())
        if k_max > 0.0:
            span_d = float(np.max(curve_xy.max(axis=0) - curve_xy.min(axis=0)))
            comb_len = 0.13 * span_d
            tips = pc - (K / k_max) * comb_len
            for j in range(0, len(uc), 3):
                ax.plot(
                    [pc[j, 0], tips[j, 0]],
                    [pc[j, 1], tips[j, 1]],
                    color=SERIES_COMB,
                    linewidth=0.5,
                    alpha=0.35,
                    zorder=3,
                )
            (h_comb,) = ax.plot(
                tips[:, 0],
                tips[:, 1],
                color=SERIES_COMB,
                linewidth=0.9,
                alpha=0.8,
                zorder=3,
                label="curvature comb",
            )
            handles.append(h_comb)

    # The spline itself.
    (h_spline,) = ax.plot(
        curve_xy[:, 0],
        curve_xy[:, 1],
        color=SERIES_SPLINE,
        linewidth=1.9,
        zorder=4,
        solid_capstyle="round",
        label="PH spline",
    )
    handles.insert(0, h_spline)

    # Input points with a surface ring so they read over the curve.
    if m + 1 <= 40:
        ms = 5.2
    elif m + 1 <= 130:
        ms = 3.2
    else:
        ms = 1.7
    (h_pts,) = ax.plot(
        pts_d[:, 0],
        pts_d[:, 1],
        linestyle="none",
        marker="o",
        markersize=ms,
        markerfacecolor=SURFACE,
        markeredgecolor=INK,
        markeredgewidth=0.9 if ms > 2 else 0.5,
        zorder=5,
        label="input points",
    )
    handles.insert(1, h_pts)

    ax.legend(handles=handles, loc="best", handlelength=1.6)

    subtitle = spline_stats(curve)
    if note:
        subtitle += f"\n{note}"
    if tr_note:
        subtitle += f"\n{tr_note}"
    n_lines = subtitle.count("\n") + 1
    fig.subplots_adjust(
        top=0.918 - 0.0285 * n_lines, left=0.10, right=0.97, bottom=0.08
    )
    fig.text(
        0.05,
        0.975,
        f"{index:02d} · {name.replace('_', ' ')}",
        fontsize=11,
        color=INK,
        ha="left",
        va="top",
    )
    fig.text(
        0.05,
        0.930,
        subtitle,
        fontsize=7.2,
        color=INK_2,
        ha="left",
        va="top",
        linespacing=1.4,
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{index:02d}_{name}.png")
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return curve


def polyline_from_turns(lengths, turns, tau=1, psi0=0.0, origin=(0.0, 0.0)):
    pts = [list(origin)]
    psi = psi0
    for k, L in enumerate(lengths):
        if k > 0:
            psi += tau * turns[k - 1]
        p = pts[-1]
        pts.append([p[0] + L * math.cos(psi), p[1] + L * math.sin(psi)])
    return pts


def circle_points(R=1.0, a0=0.0, a1=1.5, n=8, cw=False, center=(0.0, 0.0)):
    angles = np.linspace(a0, a1, n)
    if cw:
        angles = -angles
    return [[center[0] + R * math.cos(a), center[1] + R * math.sin(a)] for a in angles]


def log_spiral_points(a=1.0, b=0.12, t0=0.0, t1=4 * math.pi, n=30, cw=False):
    ts = np.linspace(t0, t1, n)
    pts = [
        [a * math.exp(b * t) * math.cos(t), a * math.exp(b * t) * math.sin(t)]
        for t in ts
    ]
    if cw:
        pts = [[p[0], -p[1]] for p in pts]
    return pts


def graph_points(fn, x0, x1, n):
    xs = np.linspace(x0, x1, n)
    return [[float(x), float(fn(x))] for x in xs]
