"""Gallery support for the main class's direct general-data construction.

Every example is built by one ``CubicPHSplineOpen``.  Alternate shades expose
the class's internally verified convex sub-splines; they are not separate
curves and their boundaries are the section-22 G1 joints, never G0 cuts.

It also provides the gallery renderer: every plot marks auxiliary inflection
points with translucent magenta crosses and 10 points that are equidistant in
arc length with red circles.  An inset strip verifies the red-point spacing
**independently** by measuring the gaps on a dense sampled polyline of the
rendered geometry - a chordal integration that never consults the package's
closed-form arc length.
"""

from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from _common import (
    BASELINE,
    INK,
    INK_2,
    SERIES_SPLINE,
    SURFACE,
    _display_transform,
)

from ph_spline import CubicPHSplineOpen

#: Categorical slot 8 (red) from the validated reference palette.
SERIES_RED = "#e34948"
#: Magenta used for translucent auxiliary-inflection crosses.
SERIES_MAGENTA = "#d946ef"
#: Sequential blue step 300: the alternating run shade.
SERIES_SPLINE_ALT = "#6da7ec"


class GalleryCurve:
    """Thin plotting adapter around one general-data ``CubicPHSplineOpen``."""

    def __init__(self, points) -> None:
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
            raise ValueError("example input must be an (n, 2) point array")
        self.points = arr
        data = [[float(x), float(y)] for x, y in arr]
        self.curve = CubicPHSplineOpen(data)
        self.total_length = self.curve.arc_length(1.0)
        self.n_segments = len(self.curve._segments)
        starts = [0]
        starts.extend(
            j + 1 for j, kind in enumerate(self.curve._joint_kinds) if kind != "g2"
        )
        self.runs = [(a, b) for a, b in zip(starts, starts[1:] + [self.n_segments])]

    def point_at_length(self, s: float) -> np.ndarray:
        """Point on the one spline at arc length ``s``."""
        s = min(max(float(s), 0.0), self.total_length)
        return self.curve.point_at_length(s)

    def sample_runs(self, budget: int = 45000) -> list[np.ndarray]:
        """Arc-dense samples of every run (user coordinates).

        Sample parameters are placed arc-uniformly within each segment so
        that extreme speed variation cannot leave coverage gaps.  The
        placement uses the segment arc-length inverse, but every returned
        coordinate is evaluated with plain de Casteljau geometry, so a
        chordal measure over these samples remains an independent check
        of the arc-length code: misplacement would appear as coverage
        gaps, not as self-consistent lengths.
        """
        ds = self.total_length / budget
        out = []
        curve = self.curve
        origin = np.asarray(curve._origin, dtype=np.float64)
        scale = curve._scale
        for a, b in self.runs:
            chunks = []
            for seg in curve._segments[a:b]:
                n = int(min(2000, max(8, math.ceil(seg.length * scale / ds))))
                s_local = np.linspace(0.0, seg.length, n + 1)
                ts = np.array(
                    [0.0]
                    + [seg.invert_arc_length_local(float(s)) for s in s_local[1:-1]]
                    + [1.0]
                )
                u = 1.0 - ts
                c = seg.ctrl
                p01 = np.outer(u, c[0]) + np.outer(ts, c[1])
                p12 = np.outer(u, c[1]) + np.outer(ts, c[2])
                p23 = np.outer(u, c[2]) + np.outer(ts, c[3])
                p012 = u[:, None] * p01 + ts[:, None] * p12
                p123 = u[:, None] * p12 + ts[:, None] * p23
                xy = u[:, None] * p012 + ts[:, None] * p123
                chunks.append(origin + scale * xy)
            out.append(np.concatenate(chunks, axis=0))
        return out


# ---------------------------------------------------------------------------
# Independent equidistance verification
# ---------------------------------------------------------------------------


def equidistance_check(
    comp: GalleryCurve, dense: np.ndarray, n_marks: int = 10
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Independently verify the red marks are arc-equidistant.

    The marks come from the package's closed-form inverse.  The check
    projects each mark onto a dense sampled polyline and measures the gaps
    with plain chordal summation - an arc-length measure that never uses
    the package's arc-length code.  Returns
    ``(marks, gaps, max_rel_deviation, polyline_length)``.
    """
    L = comp.total_length
    marks = np.array(
        [comp.point_at_length(k * L / (n_marks - 1)) for k in range(n_marks)]
    )
    seg = np.diff(dense, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    poly_L = float(cum[-1])

    positions = []
    for k in range(n_marks):
        p = marks[k]
        expected = k * poly_L / (n_marks - 1)
        # The window only disambiguates branches of self-crossing curves;
        # the gap measure itself stays purely chordal.  Any spacing error
        # below the window width shows up in the measured gaps, anything
        # above it hits the window edge and inflates the deviation.
        window = 0.015 * poly_L
        mask = np.abs(cum - expected) <= window
        if np.count_nonzero(mask) < 8:
            mask = np.ones(cum.shape[0], dtype=bool)
        idx = np.nonzero(mask)[0]
        d2 = (dense[idx, 0] - p[0]) ** 2 + (dense[idx, 1] - p[1]) ** 2
        j = int(idx[np.argmin(d2)])
        best_s = float(cum[j])
        best_d = float(np.sqrt(d2.min()))
        for a in (j - 1, j):
            if a < 0 or a + 1 >= dense.shape[0] or seg_len[a] == 0.0:
                continue
            ax, ay = dense[a]
            bx, by = dense[a + 1]
            t = ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / (seg_len[a] ** 2)
            t = min(max(t, 0.0), 1.0)
            qx = ax + t * (bx - ax)
            qy = ay + t * (by - ay)
            d = math.hypot(p[0] - qx, p[1] - qy)
            if d < best_d:
                best_d = d
                best_s = float(cum[a]) + t * float(seg_len[a])
        positions.append(best_s)
    positions_arr = np.array(positions)
    if not np.all(np.diff(positions_arr) > 0.0):
        raise AssertionError("projected mark positions are not monotone")
    gaps = np.diff(positions_arr)
    mean = float(gaps.mean())
    dev = float(np.max(np.abs(gaps - mean)) / mean)
    return marks, gaps, dev, poly_L


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def plot_nonconvex(
    index: int,
    name: str,
    points,
    out_dir: str,
    note: str = "",
    gap_tol: float = 5e-3,
) -> GalleryCurve:
    """Build, independently verify, and render one general-data spline."""
    comp = GalleryCurve(points)

    run_samples = comp.sample_runs()
    dense = np.concatenate(run_samples, axis=0)

    marks, _gaps, dev, poly_L = equidistance_check(comp, dense)
    if not dev <= gap_tol:
        raise AssertionError(
            f"equidistance deviation {dev:.3e} exceeds tolerance {gap_tol:.1e}"
        )
    L = comp.total_length
    len_agree = abs(poly_L - L) / L
    if not len_agree <= 5e-3:
        raise AssertionError(
            f"independent length {poly_L!r} disagrees with closed form {L!r}"
        )

    pts = comp.points
    n_pts = pts.shape[0]
    cx, cy, sc, tr_note = _display_transform(pts)

    def disp(a: np.ndarray) -> np.ndarray:
        return (a - np.array([cx, cy])) / sc

    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=110)

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, which="major")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = []
    if n_pts <= 400:
        pd = disp(pts)
        (h_poly,) = ax.plot(
            pd[:, 0],
            pd[:, 1],
            linestyle=(0, (4, 3)),
            linewidth=0.7,
            color=BASELINE,
            zorder=2,
            label="input polyline",
        )
        handles.append(h_poly)

    h_run = None
    for r, xy in enumerate(run_samples):
        xyd = disp(xy)
        color = SERIES_SPLINE if r % 2 == 0 else SERIES_SPLINE_ALT
        (h,) = ax.plot(
            xyd[:, 0],
            xyd[:, 1],
            color=color,
            linewidth=1.7,
            zorder=4,
            solid_capstyle="round",
        )
        if r == 0:
            h_run = h
            h.set_label("one PH spline (alt. shades = convex sub-splines)")
    if h_run is not None:
        handles.insert(0, h_run)

    if n_pts <= 400:
        pd = disp(pts)
        ms = 3.4 if n_pts <= 130 else 1.8
        (h_pts,) = ax.plot(
            pd[:, 0],
            pd[:, 1],
            linestyle="none",
            marker="o",
            markersize=ms,
            markerfacecolor=SURFACE,
            markeredgecolor=INK,
            markeredgewidth=0.6,
            zorder=5,
            label="input points",
        )
        handles.append(h_pts)

    aux_inflections = comp.curve.aux_inflection_points
    if aux_inflections:
        aux_xy = np.array(
            [[item["x"], item["y"]] for item in aux_inflections],
            dtype=np.float64,
        )
        aux_display = disp(aux_xy)
        (h_aux,) = ax.plot(
            aux_display[:, 0],
            aux_display[:, 1],
            linestyle="none",
            marker="x",
            markersize=5.0,
            color=SERIES_MAGENTA,
            markeredgewidth=0.9,
            alpha=0.5,
            zorder=7,
            label="auxiliary inflection points",
        )
        handles.append(h_aux)

    md = disp(marks)
    (h_red,) = ax.plot(
        md[:, 0],
        md[:, 1],
        linestyle="none",
        marker="o",
        markersize=7.0,
        markerfacecolor=SERIES_RED,
        markeredgecolor=SURFACE,
        markeredgewidth=1.4,
        zorder=6,
        label="10 arc-equidistant points",
    )
    handles.append(h_red)
    ax.legend(handles=handles, loc="best", handlelength=1.5)

    subtitle = (
        f"{n_pts} pts · {len(comp.runs)} G² convex sub-splines · "
        f"{comp.n_segments} segments · "
        + (f"L = {L:.4g}" if 1e-3 <= L < 1e6 else f"L = {L:.3e}")
    )
    if note:
        subtitle += f"\n{note}"
    if tr_note:
        subtitle += f"\n{tr_note}"
    n_lines = subtitle.count("\n") + 1
    fig.text(
        0.05,
        0.978,
        f"{index:02d} · {name.replace('_', ' ')}",
        fontsize=11,
        color=INK,
        ha="left",
        va="top",
    )
    fig.text(
        0.05,
        0.943,
        subtitle,
        fontsize=7.0,
        color=INK_2,
        ha="left",
        va="top",
        linespacing=1.4,
    )
    fig.text(
        0.50,
        0.018,
        "independent dense-polyline audit · "
        f"|L_dense/L_exact − 1| = {len_agree:.2e} · "
        f"max |Δs/mean(Δs) − 1| = {dev:.2e}",
        fontsize=6.4,
        color=INK_2,
        ha="center",
        va="bottom",
    )
    fig.subplots_adjust(
        top=0.918 - 0.0285 * n_lines,
        left=0.10,
        right=0.97,
        bottom=0.09,
    )

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{index:02d}_{name}.png")
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return comp


# ---------------------------------------------------------------------------
# Authoring helpers
# ---------------------------------------------------------------------------


class Turtle:
    """Single-stroke path authoring helper for straights, arcs, and turns."""

    def __init__(self, pos=(0.0, 0.0), heading=0.0) -> None:
        self.pts: list[list[float]] = [[float(pos[0]), float(pos[1])]]
        self.heading = float(heading)

    def straight(self, length: float, n: int = 1) -> Turtle:
        step = length / n
        for _ in range(n):
            x, y = self.pts[-1]
            self.pts.append(
                [x + step * math.cos(self.heading), y + step * math.sin(self.heading)]
            )
        return self

    def arc(self, radius: float, angle: float, n: int | None = None) -> Turtle:
        if n is None:
            n = max(2, math.ceil(abs(angle) / 0.30))
        side = 1.0 if angle > 0 else -1.0
        x, y = self.pts[-1]
        cxa = x + radius * side * (-math.sin(self.heading))
        cya = y + radius * side * (math.cos(self.heading))
        vx, vy = x - cxa, y - cya
        for k in range(1, n + 1):
            a = angle * k / n
            ca, sa = math.cos(a), math.sin(a)
            self.pts.append([cxa + ca * vx - sa * vy, cya + sa * vx + ca * vy])
        self.heading += angle
        return self

    def corner(self, angle: float) -> Turtle:
        self.heading += angle
        return self

    def data(self) -> list[list[float]]:
        return self.pts


def arc_pts(center, r, th0, th1, n):
    """Sampled circle arc from angle ``th0`` to ``th1``."""
    ths = np.linspace(th0, th1, n)
    return [[center[0] + r * math.cos(t), center[1] + r * math.sin(t)] for t in ths]


def param(fx, fy, t0, t1, n):
    ts = np.linspace(t0, t1, n)
    return [[float(fx(t)), float(fy(t))] for t in ts]
