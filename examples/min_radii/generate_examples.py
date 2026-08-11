"""Render 8 ``min_curvature_radii`` studies: 2 per spline variety.

Run from the repository root:
    python examples/min_radii/generate_examples.py

Every case reads the exact minimal left/right curvature radii
``(rho_left, rho_right)`` from the new property, marks the critical
curvature points with their osculating circles, and draws exact NURBS
offsets below, at, and beyond the cusp distance: the offset develops its
first cusp exactly at ``d = rho``.  Each offset is verified against
``r(u) + d * N_L(u)`` and the cusp condition ``1 - rho * kappa_max = 0``
is asserted before an image is published.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from offset_common import (  # noqa: E402
    GRID,
    INK,
    MUTED,
    PAGE,
    POINT,
    SURFACE,
    verify_offset,
)

import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ph_spline import (  # noqa: E402
    CubicPHSplineClosed,
    CubicPHSplineOpen,
    PHBSplineClosed,
    PHBSplineOpen,
)

OUT = Path(__file__).resolve().parent
SUB = "#246fc2"     # sub-critical offset
CRIT = "#c2242b"    # cusp-critical offset d = rho
BEYOND = "#b28ac2"  # beyond-critical offset with swallowtail loops
CIRCLE = "#167b6b"  # osculating circle at the critical point


def critical_point(curve, sign):
    """Parameter, point, and left normal at the extreme +-curvature."""
    uu = np.linspace(0.0, 1.0, 40001)
    kk = np.array([sign * curve.signed_curvature(float(u)) for u in uu])
    i = int(np.argmax(kk))
    lo = float(uu[max(i - 1, 0)])
    hi = float(uu[min(i + 1, uu.size - 1)])
    for _ in range(120):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if sign * curve.signed_curvature(m1) < sign * curve.signed_curvature(m2):
            lo = m1
        else:
            hi = m2
    u = 0.5 * (lo + hi)
    for candidate in (0.0, 1.0):
        if sign * curve.signed_curvature(candidate) > sign * (
            curve.signed_curvature(u)
        ):
            u = candidate
    return u, curve.point(u), curve.normal(u)


def render(index, name, curve, family, sides, note, check_only):
    rho_left, rho_right = curve.min_curvature_radii
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=130, facecolor=PAGE)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color=GRID, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)

    uu = np.linspace(0.0, 1.0, 900)
    source = np.asarray([curve.point(float(u)) for u in uu])

    labeled = set()
    for side in sides:
        rho = rho_left if side > 0 else rho_right
        assert math.isfinite(rho)
        u_star, p_star, n_star = critical_point(curve, float(side))
        kappa_star = abs(curve.signed_curvature(u_star))
        assert abs(1.0 - rho * kappa_star) <= 1e-6, (name, rho, kappa_star)
        for factor, color, style, label in (
            (0.65, SUB, "-", "offset below rho: cusp-free"),
            (1.0, CRIT, "-", "offset at d = rho: first cusp"),
            (1.3, BEYOND, "-", "offset beyond rho: swallowtail"),
        ):
            d = side * factor * rho
            handle = curve.offset(d)
            verify_offset(curve, handle, d)
            if check_only:
                continue
            samples = np.asarray([handle.point(float(u)) for u in uu])
            key = label
            ax.plot(samples[:, 0], samples[:, 1], color=color,
                    linestyle=style, linewidth=1.25, zorder=2,
                    label=None if key in labeled else label)
            labeled.add(key)
        if check_only:
            continue
        center = p_star + side * rho * n_star
        circle = plt.Circle(center, rho, fill=False, color=CIRCLE,
                            linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
        ax.add_patch(circle)
        ax.scatter(*p_star, s=42, marker="o", facecolor="white",
                   edgecolor=CIRCLE, linewidth=1.4, zorder=6,
                   label=None if "crit" in labeled else
                   "extreme-curvature point + osculating circle")
        labeled.add("crit")
    if check_only:
        print(f"verified {index:02d}_{name}")
        return

    ax.plot(source[:, 0], source[:, 1], color=INK, linewidth=2.1, zorder=4,
            label=f"{family} source")
    nodes = np.asarray(
        curve.points if hasattr(curve, "points") else curve._points
    )
    ax.scatter(nodes[:, 0], nodes[:, 1], s=11, facecolor=SURFACE,
               edgecolor=POINT, linewidth=0.7, zorder=5)

    def fmt(value):
        return "inf" if math.isinf(value) else f"{value:.4g}"

    fig.suptitle(f"{index:02d} · {name.replace('_', ' ')}", x=0.06,
                 ha="left", color=INK, fontsize=10.5)
    ax.set_title(
        f"min_curvature_radii = ({fmt(rho_left)}, {fmt(rho_right)}) · "
        f"cusp-free offsets for -rho_right < d < rho_left\n{note}",
        loc="left", color=INK, fontsize=7.2,
    )
    ax.legend(loc="best", frameon=False, fontsize=6.6)
    fig.savefig(OUT / f"{index:02d}_{name}.png", bbox_inches="tight",
                facecolor=PAGE)
    plt.close(fig)
    print(f"rendered {index:02d}_{name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    check_only = parser.parse_args().check_only

    cases = []

    # -- cubic open ------------------------------------------------------
    cases.append((
        "cubic_open_cam_flank",
        CubicPHSplineOpen(
            [[t, 0.85 * math.exp(-1.4 * (t - 2.1) ** 2)]
             for t in np.linspace(0.0, 4.2, 17)]
        ),
        "open cubic PH", (1,),
        "one-sided cam flank: the lift crest fixes the largest cusp-free "
        "clearance band above the profile",
    ))
    cases.append((
        "cubic_open_s_transition",
        CubicPHSplineOpen(
            [[0.0, 0.0], [1.0, 0.8], [2.0, 1.0], [3.0, 0.2], [4.0, 0.0],
             [5.0, 0.8]]
        ),
        "open cubic PH", (1, -1),
        "S transition: each bend bounds its own side; both critical "
        "offsets cusp exactly at their osculating radius",
    ))

    # -- cubic closed ----------------------------------------------------
    cases.append((
        "cubic_closed_pocket_limit",
        CubicPHSplineClosed(
            [[2.4 * math.cos(t) + 0.3 * math.cos(2 * t),
              1.5 * math.sin(t) + 0.15 * math.sin(2 * t)]
             for t in np.linspace(0.0, 2 * math.pi, 21, endpoint=False)]
        ),
        "closed cubic PH", (1,),
        "pocket boundary: rho_left is the deepest cusp-free inward "
        "clearing pass for a zero-radius tool",
    ))
    cases.append((
        "cubic_closed_wavy_ring",
        CubicPHSplineClosed(
            [[(2.0 + 0.28 * math.cos(4 * t)) * math.cos(t),
              (2.0 + 0.28 * math.cos(4 * t)) * math.sin(t)]
             for t in np.linspace(0.0, 2 * math.pi, 26, endpoint=False)]
        ),
        "closed cubic PH", (1, -1),
        "wavy ring: lobes bound the inward side, valleys the outward "
        "side, with cusps arriving exactly on schedule",
    ))

    # -- PH B-spline open ------------------------------------------------
    cases.append((
        "bspline_open_g4_rail",
        PHBSplineOpen(
            [[0.0, 0.0], [1.4, 0.7], [2.8, 0.5], [4.2, 1.3], [5.6, 1.0],
             [7.0, 1.7]],
            g_order=4,
        ),
        "open PH B-spline (G4)", (1, -1),
        "G4 rail alignment: smooth high-order geometry still has hard "
        "cusp budgets on both sides",
    ))
    cases.append((
        "bspline_open_duct_flank",
        PHBSplineOpen(
            [[t, 0.55 * math.sin(0.9 * t) + 0.1 * t]
             for t in np.linspace(0.0, 6.0, 13)]
        ),
        "open PH B-spline", (1,),
        "duct flank: the tightest left bend caps the one-sided "
        "insulation build-up",
    ))

    # -- PH B-spline closed ----------------------------------------------
    cases.append((
        "bspline_closed_impeller_ring",
        PHBSplineClosed(
            [[(1.8 + 0.24 * math.cos(3 * t)) * math.cos(t),
              (1.8 + 0.24 * math.cos(3 * t)) * math.sin(t)]
             for t in np.linspace(0.0, 2 * math.pi, 18, endpoint=False)]
        ),
        "closed PH B-spline", (1,),
        "impeller ring: the lobe tips set the deepest cusp-free "
        "inward shroud clearance",
    ))
    cases.append((
        "bspline_closed_g3_wave_ring",
        PHBSplineClosed(
            [[(2.0 + 0.3 * math.cos(5 * t)) * math.cos(t),
              (2.0 + 0.3 * math.cos(5 * t)) * math.sin(t)]
             for t in np.linspace(0.0, 2 * math.pi, 24, endpoint=False)],
            g_order=3,
        ),
        "closed PH B-spline (G3)", (1, -1),
        "G3 wave ring: lobe crests bound the inward side, troughs the "
        "outward side",
    ))

    for index, (name, curve, family, sides, note) in enumerate(cases, 1):
        render(index, name, curve, family, sides, note, check_only)
    print(f"\nAll {len(cases)} cases "
          f"{'verified' if check_only else 'rendered'} into {OUT}")


if __name__ == "__main__":
    main()
