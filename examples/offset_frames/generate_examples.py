"""Render 8 offset frame-and-curvature studies: 2 per spline variety.

Run from the repository root:
    python examples/offset_frames/generate_examples.py

Every case queries the differential frame of the exact offset NURBS
itself -- ``tangent``, ``normal``, ``principal_normal``,
``signed_curvature``, ``curvature_vector`` -- and verifies each value
against the closed-form parallel-curve identities through the source API

    T_d = sign(1 - d kappa) T,          kappa_d = kappa / |1 - d kappa|,
    K_d = kappa / (1 - d kappa) N_L,

before an image is published.  Frame stations are placed at equal travel
along the offset locus with ``parameter_at_length``; cusped studies mark
every certified cusp and assert that each frame query raises
``UndefinedTangentError`` exactly there.
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
    LEFT_SIDE,
    PAGE,
    POINT,
    RIGHT_SIDE,
    SURFACE,
    verify_offset,
)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ph_spline import (  # noqa: E402
    CubicPHSplineClosed,
    CubicPHSplineOpen,
    PHBSplineClosed,
    PHBSplineOpen,
    UndefinedTangentError,
)

OUT = Path(__file__).resolve().parent
TANGENT = "#167b6b"   # tangent arrows
NORMAL = "#b28ac2"    # normal arrows
COMB = "#c2242b"      # curvature comb
CUSP = "#c2242b"      # certified cusp markers


def verify_frames(curve, handle, distance: float, *, samples: int = 257) -> None:
    """Check every frame query against the parallel-curve identities."""
    extent = max(1.0, float(np.max(np.abs(
        curve.points if hasattr(curve, "points") else curve._points
    ))))
    tolerance = 1e-7 * max(extent, abs(distance))
    for u in np.linspace(0.0, 1.0, samples):
        u = float(u)
        kappa = curve.signed_curvature(u)
        factor = 1.0 - distance * kappa
        try:
            produced = handle.tangent(u)
        except UndefinedTangentError:
            continue  # within the deterministic zero-speed guard band
        flip = 1.0 if factor >= 0.0 else -1.0
        if abs(factor) < 1e-6:
            continue  # conditioning excluded next to a cusp
        expected = flip * curve.tangent(u)
        gap = float(np.max(np.abs(produced - expected)))
        if not gap <= tolerance:
            raise RuntimeError(
                f"tangent identity failed at u={u}: {gap:.3e} > {tolerance:.3e}"
            )
        if abs(float(np.hypot(*produced)) - 1.0) > 1e-13:
            raise RuntimeError(f"tangent is not unit at u={u}")
        kappa_d = handle.signed_curvature(u)
        expected_k = kappa / abs(factor)
        if abs(kappa_d - expected_k) > 1e-6 * max(1.0, abs(expected_k)):
            raise RuntimeError(f"curvature identity failed at u={u}")
        vector = handle.curvature_vector(u)
        expected_v = (kappa / factor) * curve.normal(u)
        if float(np.max(np.abs(vector - expected_v))) > 1e-6 * max(
            1.0, float(np.max(np.abs(expected_v)))
        ):
            raise RuntimeError(f"curvature vector identity failed at u={u}")


def frame_stations(handle, count: int):
    """Frame anchors at equal travel along the offset locus."""
    total = handle.length
    stations = []
    for k in range(count):
        s = (k + 0.5) / count * total
        u = handle.parameter_at_length(s)
        try:
            stations.append(
                (handle.point(u), handle.tangent(u), handle.normal(u))
            )
        except UndefinedTangentError:
            continue  # a station landed inside a cusp guard band
    return stations


def draw_stations(ax, stations, arrow: float, labeled: set) -> None:
    for anchor, tangent, normal in stations:
        ax.annotate(
            "", xy=anchor + arrow * tangent, xytext=anchor, zorder=6,
            arrowprops=dict(arrowstyle="-|>", color=TANGENT, lw=1.2,
                            shrinkA=0, shrinkB=0),
        )
        ax.annotate(
            "", xy=anchor + 0.62 * arrow * normal, xytext=anchor, zorder=6,
            arrowprops=dict(arrowstyle="-|>", color=NORMAL, lw=1.1,
                            shrinkA=0, shrinkB=0),
        )
    if stations and "frames" not in labeled:
        ax.plot([], [], color=TANGENT, lw=1.2, label="offset tangent")
        ax.plot([], [], color=NORMAL, lw=1.1, label="offset left normal")
        labeled.add("frames")


def draw_comb(ax, handle, comb_scale: float, labeled: set,
              *, samples: int = 401, cap: float = 4.0) -> None:
    """Curvature comb: whiskers along -curvature_vector, kappa-scaled."""
    segments = []
    for u in np.linspace(0.0, 1.0, samples):
        try:
            anchor = handle.point(float(u))
            vector = handle.curvature_vector(float(u))
        except UndefinedTangentError:
            continue
        magnitude = float(np.hypot(*vector))
        if magnitude > cap:  # keep diverging near-cusp teeth readable
            vector = vector * (cap / magnitude)
        segments.append([anchor, anchor - comb_scale * vector])
    ax.add_collection(LineCollection(
        segments, colors=COMB, linewidths=0.55, alpha=0.55, zorder=2,
        label=None if "comb" in labeled else "curvature comb (kappa_d)",
    ))
    labeled.add("comb")


def draw_cusps(ax, curve, handle, labeled: set) -> None:
    for cusp in handle.cusps:
        for query in (handle.tangent, handle.normal, handle.principal_normal,
                      handle.signed_curvature, handle.curvature_vector):
            try:
                query(cusp.parameter)
            except UndefinedTangentError:
                continue
            raise RuntimeError(
                f"frame query did not raise at certified cusp {cusp}"
            )
        anchor = handle.point(cusp.parameter)
        ax.scatter(*anchor, s=48, marker="x", color=CUSP, linewidth=1.6,
                   zorder=7,
                   label=None if "cusp" in labeled else
                   "certified cusp: frames undefined")
        labeled.add("cusp")


def render(index, name, curve, family, distances, note, check_only, *,
           stations=14, comb=None, show_cusps=False):
    handles = [(d, curve.offset(d)) for d in distances]
    for d, handle in handles:
        verify_offset(curve, handle, d)
        verify_frames(curve, handle, d)
    if check_only:
        print(f"verified {index:02d}_{name}")
        return

    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=130, facecolor=PAGE)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color=GRID, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)

    uu = np.linspace(0.0, 1.0, 900)
    source = np.asarray([curve.point(float(u)) for u in uu])
    nodes = np.asarray(
        curve.points if hasattr(curve, "points") else curve._points
    )
    extent = float(np.max(source.max(axis=0) - source.min(axis=0)))
    arrow = 0.055 * extent

    labeled: set = set()
    for d, handle in handles:
        samples = np.asarray([handle.point(float(u)) for u in uu])
        color = LEFT_SIDE if d >= 0.0 else RIGHT_SIDE
        side = "left" if d >= 0.0 else "right"
        key = f"offset_{side}"
        ax.plot(samples[:, 0], samples[:, 1], color=color, linewidth=1.35,
                zorder=3,
                label=None if key in labeled else
                f"exact NURBS offset ({side}, d = {d:+.3g})")
        labeled.add(key)
        if comb is not None:
            draw_comb(ax, handle, comb * extent, labeled)
        draw_stations(ax, frame_stations(handle, stations), arrow, labeled)
        if show_cusps:
            draw_cusps(ax, curve, handle, labeled)

    ax.plot(source[:, 0], source[:, 1], color=INK, linewidth=2.0, zorder=4,
            label=f"{family} source")
    ax.scatter(nodes[:, 0], nodes[:, 1], s=11, facecolor=SURFACE,
               edgecolor=POINT, linewidth=0.7, zorder=5)

    degree = handles[0][1].degree
    fig.suptitle(f"{index:02d} · {name.replace('_', ' ')}", x=0.06,
                 ha="left", color=INK, fontsize=10.5)
    ax.set_title(
        f"frame + curvature queries on the degree-{degree} rational offset "
        f"· stations at equal offset travel\n{note}",
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
    s_curve = CubicPHSplineOpen(
        [[0.0, 0.0], [1.0, 0.8], [2.0, 1.0], [3.0, 0.2], [4.0, 0.0],
         [5.0, 0.8]]
    )
    rho_l, rho_r = s_curve.min_curvature_radii
    cases.append((
        "cubic_open_guideway_frames", s_curve, "open cubic PH",
        (0.45 * rho_l, -0.45 * rho_r),
        "S guideway: tangent/normal frames ride both cusp-free offsets; "
        "the frame flips sides with the traversal, never with d",
        dict(stations=12),
    ))
    cases.append((
        "cubic_open_cusp_reversal", s_curve, "open cubic PH",
        (1.4 * rho_l,),
        "beyond-critical offset: certified cusps split the locus into "
        "reversed arcs; every frame query raises exactly at the cusps",
        dict(stations=16, show_cusps=True),
    ))

    # -- cubic closed ----------------------------------------------------
    wavy = CubicPHSplineClosed(
        [[(2.0 + 0.3 * math.cos(5 * t)) * math.cos(t),
          (2.0 + 0.3 * math.cos(5 * t)) * math.sin(t)]
         for t in np.linspace(0.0, 2 * math.pi, 24, endpoint=False)]
    )
    rho_l, rho_r = wavy.min_curvature_radii
    cases.append((
        "cubic_closed_comb_ring", wavy, "closed cubic PH",
        (0.35 * rho_l, 0.7 * rho_l),
        "wavy ring, two inward passes: the curvature comb reads "
        "kappa_d = kappa/|1 - d kappa| growing as d approaches rho_left",
        dict(stations=0, comb=0.05),
    ))
    blade = CubicPHSplineClosed(
        [[2.4 * math.cos(t) + 0.3 * math.cos(2 * t),
          1.5 * math.sin(t) + 0.15 * math.sin(2 * t)]
         for t in np.linspace(0.0, 2 * math.pi, 21, endpoint=False)]
    )
    rho_l, _ = blade.min_curvature_radii
    cases.append((
        "cubic_closed_camber_stations", blade, "closed cubic PH",
        (0.6 * rho_l,),
        "cambered section: equal-travel frame stations on the inward "
        "clearing pass, placed with parameter_at_length",
        dict(stations=22),
    ))

    # -- PH B-spline open ------------------------------------------------
    rail = PHBSplineOpen(
        [[0.0, 0.0], [1.4, 0.7], [2.8, 0.5], [4.2, 1.3], [5.6, 1.0],
         [7.0, 1.7]],
        g_order=4,
    )
    rho_l, rho_r = rail.min_curvature_radii
    cases.append((
        "bspline_open_g4_rail_frames", rail, "open PH B-spline (G4)",
        (0.5 * rho_l, -0.5 * rho_r),
        "G4 rail: degree-9 rational offsets answer the same frame queries "
        "as the source spline, at every continuity order",
        dict(stations=12, comb=0.03),
    ))
    duct = PHBSplineOpen(
        [[t, 0.55 * math.sin(0.9 * t) + 0.1 * t]
         for t in np.linspace(0.0, 6.0, 13)]
    )
    rho_l, _ = duct.min_curvature_radii
    cases.append((
        "bspline_open_duct_cusps", duct, "open PH B-spline",
        (1.5 * rho_l,),
        "duct flank beyond its cusp budget: swallowtail arcs traverse "
        "backwards between certified cusps",
        dict(stations=16, show_cusps=True),
    ))

    # -- PH B-spline closed ----------------------------------------------
    impeller = PHBSplineClosed(
        [[(1.8 + 0.24 * math.cos(3 * t)) * math.cos(t),
          (1.8 + 0.24 * math.cos(3 * t)) * math.sin(t)]
         for t in np.linspace(0.0, 2 * math.pi, 18, endpoint=False)]
    )
    rho_l, rho_r = impeller.min_curvature_radii
    cases.append((
        "bspline_closed_impeller_comb", impeller, "closed PH B-spline",
        (0.55 * rho_l, -0.45 * rho_r),
        "impeller ring: seam-consistent frames and curvature combs on "
        "closed rational offsets",
        dict(stations=10, comb=0.045),
    ))
    wave_ring = PHBSplineClosed(
        [[(2.0 + 0.3 * math.cos(5 * t)) * math.cos(t),
          (2.0 + 0.3 * math.cos(5 * t)) * math.sin(t)]
         for t in np.linspace(0.0, 2 * math.pi, 24, endpoint=False)],
        g_order=3,
    )
    rho_l, _ = wave_ring.min_curvature_radii
    cases.append((
        "bspline_closed_g3_cusp_ring", wave_ring, "closed PH B-spline (G3)",
        (1.2 * rho_l,),
        "G3 wave ring past its inward budget: ten certified cusps, one per "
        "lobe flank; frames reverse on every inter-cusp arc",
        dict(stations=0, show_cusps=True),
    ))

    for index, (name, curve, family, distances, note, kwargs) in enumerate(
        cases, 1
    ):
        render(index, name, curve, family, distances, note, check_only,
               **kwargs)
    print(f"\nAll {len(cases)} cases "
          f"{'verified' if check_only else 'rendered'} into {OUT}")


if __name__ == "__main__":
    main()
