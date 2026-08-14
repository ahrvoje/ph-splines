"""Render 128 verified closed-spline area studies into this folder.

Run from the repository root:
    python examples/closed_area/generate_examples.py

Use ``--check-only`` to compile and verify the complete corpus without
drawing.  The corpus covers all four closed varieties of the package:

- 001-032  ``CubicPHSplineClosed`` base curves;
- 033-064  exact closed offsets of cubic sources (``ClosedNURBSHandle``);
- 065-096  ``PHBSplineClosed`` base curves across continuity orders;
- 097-128  exact closed offsets of PH B-spline sources.

Every base case re-verifies its published ``signed_area`` against an
independent exact rational power-basis oracle before an image is written.
Every offset case re-verifies the parallel-curve identity
``A_d = A_0 - d L_0 + pi nu d**2`` with the exact captured source length
and the certified turning number; ``d == 0`` cases must match the source
area bitwise.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ph_spline import (
    ConstructionPolicy,
    CubicPHSplineClosed,
    NumericalPolicy,
    NumericalPrecisionError,
    PHBSplineClosed,
)
import ph_spline.area as area_module
from ph_spline.area import turning_number

OUT = Path(__file__).resolve().parent
PAGE = "#f9f9f7"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK = "#172033"
MUTED = "#8a94a6"
POINT = "#eb6834"
LEFT_SIDE = "#246fc2"
RIGHT_SIDE = "#c45b31"
FILL_POS = "#2e8b6f"
FILL_NEG = "#c45b31"


# ---------------------------------------------------------------------------
# Independent exact area oracle (power basis, Fractions)
# ---------------------------------------------------------------------------


def _power(controls):
    degree = len(controls) - 1
    out = []
    for k in range(degree + 1):
        acc = Fraction(0)
        for j in range(k + 1):
            term = comb(k, j) * controls[j]
            acc += term if (k - j) % 2 == 0 else -term
        out.append(comb(degree, k) * acc)
    return out


def _pmul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def _pder(p):
    return [k * p[k] for k in range(1, len(p))] or [Fraction(0)]


def exact_chain_area(spans, scale):
    total = Fraction(0)
    for controls in spans:
        xs = _power([Fraction(float(r[0])) for r in controls])
        ys = _power([Fraction(float(r[1])) for r in controls])
        integrand = [
            u - v
            for u, v in zip(_pmul(xs, _pder(ys)), _pmul(ys, _pder(xs)))
        ]
        total += sum(
            (c / (2 * (k + 1)) for k, c in enumerate(integrand)), Fraction(0)
        )
    count = len(spans)
    for index in range(count):
        end, start = spans[index][-1], spans[(index + 1) % count][0]
        total += (
            Fraction(float(end[0])) * Fraction(float(start[1]))
            - Fraction(float(end[1])) * Fraction(float(start[0]))
        ) / 2
    return Fraction(scale) * Fraction(scale) * total


def verify_base(curve) -> None:
    if isinstance(curve, CubicPHSplineClosed):
        spans = [segment.ctrl for segment in curve._segments]
        scale = curve._scale
    else:
        spans = [
            np.column_stack((span.position.real, span.position.imag))
            for span in curve._state.spans
        ]
        scale = float(curve._state.scale)
    oracle = float(exact_chain_area(spans, scale))
    if curve.signed_area != oracle:
        raise RuntimeError(
            f"signed_area {curve.signed_area!r} != exact oracle {oracle!r}"
        )
    if curve.area != abs(curve.signed_area):
        raise RuntimeError("area != abs(signed_area)")


def verify_offset(source, handle, distance: float) -> int:
    if distance == 0.0:
        if handle.signed_area != source.signed_area:
            raise RuntimeError("d == 0 offset area is not bitwise the source")
        return 1 if source.signed_area >= 0 else -1
    metric = handle._area_provenance.metric
    spans, H, _, _ = metric.exact_source_state()
    length = float(area_module._exact_source_length(spans, H))
    nu = turning_number(metric)
    predicted = (
        source.signed_area - distance * length
        + math.pi * nu * distance * distance
    )
    tolerance = 1.0e-9 * max(1.0, abs(predicted))
    if abs(handle.signed_area - predicted) > tolerance:
        raise RuntimeError(
            f"offset area {handle.signed_area!r} misses identity "
            f"{predicted!r} at d={distance}"
        )
    return nu


# ---------------------------------------------------------------------------
# Point-set builders
# ---------------------------------------------------------------------------


def circle(n=24, radius=1.0, center=(0.0, 0.0), phase=0.0, cw=False):
    sign = -1.0 if cw else 1.0
    return [
        [
            center[0] + radius * math.cos(sign * (2 * math.pi * k / n + phase)),
            center[1] + radius * math.sin(sign * (2 * math.pi * k / n + phase)),
        ]
        for k in range(n)
    ]


def ellipse(a, b, n=24, phase=0.013):
    return [
        [a * math.cos(2 * math.pi * k / n + phase),
         b * math.sin(2 * math.pi * k / n + phase)]
        for k in range(n)
    ]


def polar(radius_fn, n, phase=0.017):
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n + phase
        r = radius_fn(t)
        out.append([r * math.cos(t), r * math.sin(t)])
    return out


def superellipse(power, a, b, n=32):
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n + 0.011
        c, s = math.cos(t), math.sin(t)
        denom = (abs(c) ** power + abs(s) ** power) ** (1.0 / power)
        out.append([a * c / denom, b * s / denom])
    return out


def stadium(nline=5, narc=8, half=2.0, radius=1.0):
    pts = []
    for i in range(1, nline + 1):
        pts.append([-half + 2 * half * i / (nline + 1), -radius])
    for k in range(narc + 1):
        t = -math.pi / 2 + math.pi * k / narc
        pts.append([half + radius * math.cos(t), radius * math.sin(t)])
    for i in range(1, nline + 1):
        pts.append([half - 2 * half * i / (nline + 1), radius])
    for k in range(narc + 1):
        t = math.pi / 2 + math.pi * k / narc
        pts.append([-half + radius * math.cos(t), radius * math.sin(t)])
    return pts


def rounded_rectangle(w=3.0, h=1.8, r=0.5, nline=3, narc=5):
    cx, cy = w / 2 - r, h / 2 - r
    pts = []
    corners = [
        (cx, cy, 0.0), (-cx, cy, math.pi / 2),
        (-cx, -cy, math.pi), (cx, -cy, 3 * math.pi / 2),
    ]
    for (px, py, a0), (qx, qy, _) in zip(corners, corners[1:] + corners[:1]):
        for k in range(narc + 1):
            t = a0 + (math.pi / 2) * k / narc
            pts.append([px + r * math.cos(t), py + r * math.sin(t)])
        ex = px + r * math.cos(a0 + math.pi / 2)
        ey = py + r * math.sin(a0 + math.pi / 2)
        sx = qx + r * math.cos(a0)
        sy = qy + r * math.sin(a0)
        for i in range(1, nline + 1):
            u = i / (nline + 1)
            pts.append([ex + u * (sx - ex), ey + u * (sy - ey)])
    return pts


def figure_eight(n=32, phase=0.05, scale=1.0):
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n + phase
        out.append([scale * math.sin(t), scale * math.sin(t) * math.cos(t)])
    return out


def bernoulli_lemniscate(n=36, phase=0.06):
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n + phase
        denom = 1.0 + math.sin(t) ** 2
        out.append([1.4 * math.cos(t) / denom,
                    1.4 * math.sin(t) * math.cos(t) / denom])
    return out


def limacon(n=48, phase=0.03):
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n + phase
        r = 0.5 + math.cos(t)
        out.append([r * math.cos(t), r * math.sin(t)])
    return out


def convex_hull_points(seed, count=40, keep=12):
    rng = np.random.default_rng(seed)
    cloud = rng.uniform(-1.2, 1.2, size=(count, 2))
    center = cloud.mean(axis=0)
    angles = np.arctan2(cloud[:, 1] - center[1], cloud[:, 0] - center[0])
    order = np.argsort(angles)
    hull = []
    for index in order:
        candidate = cloud[index]
        hull.append(candidate)
    hull = np.asarray(hull)
    from scipy.spatial import ConvexHull

    vertices = hull[ConvexHull(hull).vertices]
    step = max(1, len(vertices) // keep)
    return vertices[::step].tolist()


def catmull_blob(seed, keys=8, samples=4, spread=1.2):
    rng = np.random.default_rng(seed)
    radii = rng.uniform(0.7, spread, size=keys)
    angles = np.linspace(0.0, 2 * math.pi, keys, endpoint=False)
    knots = [
        (r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, angles)
    ]
    values = np.asarray(knots)
    count = len(values)
    out = []
    for index in range(count):
        p0 = values[(index - 1) % count]
        p1 = values[index]
        p2 = values[(index + 1) % count]
        p3 = values[(index + 2) % count]
        for sample in range(samples):
            u = sample / samples
            point = 0.5 * (
                2.0 * p1
                + (-p0 + p2) * u
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u**2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u**3
            )
            out.append(point.tolist())
    return out


def geometric_circle(n=14, ratio=1.35, radius=1.5):
    angles = [0.0]
    step = 2 * math.pi * (ratio - 1.0) / (ratio**n - 1.0)
    for _ in range(n - 1):
        angles.append(angles[-1] + step)
        step *= ratio
    return [[radius * math.cos(a), radius * math.sin(a)] for a in angles]


# ---------------------------------------------------------------------------
# Case model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    name: str
    family: str  # cubic | cubic_offset | bspline | bspline_offset
    points: list
    note: str
    distances: tuple = ()
    kwargs: dict = field(default_factory=dict)
    edit: str | None = None  # None | "move" | "insert" | "transaction"
    display_scale: float = 1.0
    fill: bool = True


def build_curve(case: Case):
    if case.family.startswith("cubic"):
        curve = CubicPHSplineClosed(case.points)
    else:
        curve = PHBSplineClosed(case.points, **case.kwargs)
        # Edit demonstrations commit with a global repair: the strict-local
        # closed patch is verified regular and G2 but can carry micro-kinks
        # of near-zero curvature radius, which would dominate any offset
        # drawn afterwards.
        if case.edit == "move":
            index = 1
            target = np.asarray(case.points[index]) * 1.05
            curve.move_point(index, target.tolist(), repair="global")
        elif case.edit == "insert":
            a = np.asarray(case.points[2])
            b = np.asarray(case.points[3])
            curve.insert_point(
                3, (0.5 * (a + b) * 1.04).tolist(), repair="global"
            )
        elif case.edit == "transaction":
            with curve.edit(repair="global") as edit:
                edit.move_point(0, (np.asarray(case.points[0]) * 1.06).tolist())
                edit.move_point(4, (np.asarray(case.points[4]) * 0.95).tolist())
    return curve


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _sample(target, count=900):
    grid = np.linspace(0.0, 1.0, count)
    return np.asarray([target.point(float(u)) for u in grid])


def _format_area(value: float) -> str:
    return f"{value:.9g}"


def _fill_text(target) -> str:
    """Certified fill area, or an honest marker for a typed failure."""
    try:
        return _format_area(target.fill_area)
    except NumericalPrecisionError:
        return "unresolved"


def render(index: int, case: Case, check_only: bool) -> None:
    curve = build_curve(case)
    verify_base(curve)
    handles = []
    for distance in case.distances:
        handle = curve.offset(distance)
        nu = verify_offset(curve, handle, distance)
        handles.append((distance, handle, nu))
    if check_only:
        return

    ds = case.display_scale
    source = _sample(curve) / ds
    # Committed interpolation nodes (edits can move, add or delete them).
    if isinstance(curve, PHBSplineClosed):
        points = np.asarray(curve.points, dtype=np.float64) / ds
    else:
        points = np.asarray(case.points, dtype=np.float64) / ds

    fig, ax = plt.subplots(figsize=(5.8, 4.7), dpi=105, facecolor=PAGE)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color=GRID, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)

    if case.fill and not handles:
        fill_color = FILL_POS if curve.signed_area >= 0.0 else FILL_NEG
        ax.fill(source[:, 0], source[:, 1], color=fill_color, alpha=0.07,
                zorder=1)

    magnitudes = sorted({abs(d) for d, _, _ in handles if d != 0.0})
    for distance, handle, _ in handles:
        samples = _sample(handle) / ds
        base = LEFT_SIDE if distance >= 0.0 else RIGHT_SIDE
        rgb = matplotlib.colors.to_rgb(base)
        rank = (
            magnitudes.index(abs(distance)) / max(1, len(magnitudes) - 1)
            if len(magnitudes) > 1 and distance != 0.0
            else 0.0
        )
        blend = 0.1 + 0.45 * rank
        color = tuple((1.0 - blend) * c + blend for c in rgb)
        ax.plot(samples[:, 0], samples[:, 1], color=color, linewidth=1.35,
                zorder=2)

    ax.plot(source[:, 0], source[:, 1], color=INK, linewidth=2.0, zorder=3)
    step = max(1, math.ceil(len(points) / 40))
    shown = points[::step]
    ax.scatter(shown[:, 0], shown[:, 1], s=11, facecolor=SURFACE,
               edgecolor=POINT, linewidth=0.7, zorder=4)

    family_label = {
        "cubic": "CubicPHSplineClosed",
        "cubic_offset": "CubicPHSplineClosed + exact offsets",
        "bspline": "PHBSplineClosed",
        "bspline_offset": "PHBSplineClosed + exact offsets",
    }[case.family]
    lines = [
        f"signed_area = {_format_area(curve.signed_area)}"
        f" · area = {_format_area(curve.area)}"
        f" · fill_area = {_fill_text(curve)}"
    ]
    for distance, handle, nu in handles:
        cusp_note = (
            f", {len(handle.cusps)} cusps" if handle.cusps else ""
        )
        lines.append(
            f"d = {distance:+g}: signed_area = "
            f"{_format_area(handle.signed_area)}"
            f" · fill = {_fill_text(handle)} (nu = {nu}{cusp_note})"
        )
    scale_note = (
        f" · true scale x{case.display_scale:g}" if ds != 1.0 else ""
    )
    fig.suptitle(f"{index:03d} · {case.name.replace('_', ' ')}", x=0.06,
                 y=0.98, ha="left", color=INK, fontsize=10.5)
    subtitle = f"{family_label} · {curve.num_points} nodes{scale_note}"
    if case.note:
        subtitle += f"\n{case.note}"
    ax.set_title(subtitle, loc="left", color=MUTED, fontsize=7.0, pad=7)
    # Area readouts live in a caption below the axes, clear of the titles,
    # the tick labels, and the geometry itself.
    ax.annotate(
        "\n".join(lines),
        xy=(0.0, 0.0), xycoords="axes fraction",
        xytext=(0.0, -30.0), textcoords="offset points",
        ha="left", va="top", color=INK, fontsize=6.8, linespacing=1.5,
    )
    fig.savefig(OUT / f"{index:03d}_{case.name}.png",
                bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# The 128-case corpus
# ---------------------------------------------------------------------------


def all_cases() -> list[Case]:
    cases: list[Case] = []

    def add(name, family, points, note, **kw):
        cases.append(Case(name=name, family=family, points=points,
                          note=note, **kw))

    # -- 001-032: cubic closed base curves ------------------------------
    add("unit_circle_ccw", "cubic", circle(24),
        "Strictly convex cyclic G2 solve; area slightly above pi r^2.")
    add("unit_circle_cw", "cubic", circle(24, cw=True),
        "Clockwise traversal: signed_area is negative, area is not.")
    add("triangle_minimum", "cubic", circle(3, radius=2.0),
        "Smallest accepted closed point count.")
    add("dense_circle_200", "cubic", circle(200),
        "Many-segment convex cycle; O(S) first query.")
    add("ellipse_2to1", "cubic", ellipse(2.0, 1.0),
        "Convex ellipse: area near pi a b.")
    add("ellipse_aspect_20", "cubic", ellipse(5.0, 0.25, n=48),
        "High-aspect convex cycle.")
    add("egg_profile", "cubic",
        polar(lambda t: 1.0 / (1.0 + 0.18 * math.cos(t)), 28),
        "Egg-like convex oval.")
    add("squircle", "cubic", superellipse(4.0, 1.4, 1.4),
        "Superellipse exponent 4.")
    add("soft_diamond", "cubic", superellipse(1.6, 1.5, 1.2),
        "Superellipse exponent 1.6.")
    add("star4_wave", "cubic",
        polar(lambda t: 1.0 + 0.35 * math.cos(4 * t), 16),
        "Nonconvex general cyclic path with auxiliary inflections.")
    add("star5_wave", "cubic",
        polar(lambda t: 1.0 + 0.25 * math.cos(5 * t), 24),
        "Five-lobed nonconvex cycle.")
    add("gear_8_lobes", "cubic",
        polar(lambda t: 1.0 + 0.12 * math.cos(8 * t), 48),
        "Eight shallow lobes; sixteen curvature-sign changes.")
    add("stadium_straight_runs", "cubic", stadium(),
        "Straight runs with exactly zero PH curvature spans.")
    add("rounded_rectangle", "cubic", rounded_rectangle(),
        "Straight/curved transitions at every corner.")
    add("kidney", "cubic",
        polar(lambda t: 0.9 + 0.35 * math.cos(t) + 0.18 * math.cos(2 * t), 30),
        "Asymmetric nonconvex closed profile.")
    add("cassini_peanut", "cubic",
        polar(lambda t: math.sqrt(1.0 + 0.8 * math.cos(2 * t)), 40),
        "Cassini-style waisted oval.")
    add("dimpled_limacon", "cubic",
        polar(lambda t: 1.0 + 0.6 * math.cos(t), 32),
        "Dimpled limacon (simple, b/a = 0.6).")
    add("random_convex_hull", "cubic", convex_hull_points(11),
        "Seeded random convex hull vertices.")
    add("teardrop", "cubic",
        polar(lambda t: 1.0 - 0.55 * math.sin(t), 30),
        "Teardrop-like dimpled profile.")
    add("cam_profile", "cubic",
        polar(lambda t: 1.0 + 0.3 * math.cos(t) + 0.12 * math.cos(2 * t)
              + 0.05 * math.cos(3 * t), 36),
        "Composite-harmonic cam lobe.")
    add("wankel_trilobe", "cubic",
        polar(lambda t: 1.0 + 0.18 * math.cos(3 * t), 30),
        "Three-lobed epitrochoid-like profile.")
    add("seam_rotated_circle", "cubic", circle(16)[5:] + circle(16)[:5],
        "Cyclic seam moved five nodes; area is seam invariant.")
    add("clockwise_star", "cubic",
        polar(lambda t: 1.0 + 0.35 * math.cos(4 * t), 16)[::-1],
        "Reversed traversal negates signed_area only.")
    add("tiny_scale_1e150", "cubic", circle(12, radius=1.0e-150),
        "Area near 3.1e-300; normalized-frame arithmetic.",
        display_scale=1.0e-150)
    add("huge_scale_1e100", "cubic", circle(12, radius=1.0e100),
        "Area near 3.1e200 without overflow.", display_scale=1.0e100)
    add("far_translated_1e9", "cubic", circle(12, center=(1.0e9, -1.0e9)),
        "Unit shape one billion units away; translation invariant.")
    add("geometric_spacing", "cubic", geometric_circle(),
        "Chord lengths in geometric progression.")
    add("flat_ellipse_100", "cubic", ellipse(10.0, 0.1, n=64),
        "Aspect ratio 100 convex cycle.")
    add("wavy_convex", "cubic",
        polar(lambda t: 1.0 + 0.05 * math.sin(6 * t)
              + 0.03 * math.cos(11 * t), 44),
        "Slightly wavy convex-ish blob.")
    add("octagon", "cubic",
        [[1.2, 0.0], [0.9, 0.9], [0.0, 1.2], [-0.9, 0.9], [-1.2, 0.0],
         [-0.9, -0.9], [0.0, -1.2], [0.9, -0.9]],
        "Regular octagon nodes.")
    add("crescent_bay", "cubic",
        polar(lambda t: 1.0 - 0.45 * math.cos(t) + 0.1 * math.cos(2 * t), 34),
        "Deeply dimpled bay profile.")
    add("grand_composite_blob", "cubic", catmull_blob(29, keys=10, samples=4),
        "Seeded Catmull-Rom design blob.")

    # -- 033-064: cubic closed offsets ----------------------------------
    add("circle_inward_fan", "cubic_offset", circle(24),
        "Left-normal offsets of a CCW circle go inward.",
        distances=(0.2, 0.4, 0.6))
    add("circle_outward_fan", "cubic_offset", circle(24),
        "Negative distances offset outward.",
        distances=(-0.25, -0.5, -0.75))
    add("circle_zero_distance", "cubic_offset", circle(24),
        "d = 0: offset area equals the source area bitwise.",
        distances=(0.0,))
    add("ellipse_cusp_onset", "cubic_offset", ellipse(2.0, 1.0),
        "Distance beyond the smallest curvature radius forms cusps.",
        distances=(0.55,))
    add("circle_reversed_loop", "cubic_offset", circle(24),
        "d beyond every radius: reversed loop, area near pi (d - r)^2.",
        distances=(1.6,))
    add("cw_circle_both_sides", "cubic_offset", circle(24, cw=True),
        "Clockwise source with both offset signs.",
        distances=(0.25, -0.25))
    add("star4_cusped", "cubic_offset",
        polar(lambda t: 1.0 + 0.35 * math.cos(4 * t), 16),
        "Cusp-forming offset of a nonconvex star.", distances=(0.35,))
    add("star5_both_sides", "cubic_offset",
        polar(lambda t: 1.0 + 0.25 * math.cos(5 * t), 24),
        "Cusp-free two-sided offsets.", distances=(0.15, -0.15))
    add("stadium_shells", "cubic_offset", stadium(),
        "Offsets across straight and curved spans.",
        distances=(0.15, -0.15))
    add("stadium_deep", "cubic_offset", stadium(),
        "Deep interior offset of the stadium.", distances=(0.45,))
    add("ellipse_bands", "cubic_offset", ellipse(3.0, 1.2, n=32),
        "Nested cusp-free interior bands.", distances=(0.2, 0.4))
    add("kidney_cusps", "cubic_offset",
        polar(lambda t: 0.9 + 0.35 * math.cos(t) + 0.18 * math.cos(2 * t), 30),
        "Cusped interior offset of a kidney profile.", distances=(0.3,))
    add("dimpled_limacon_offsets", "cubic_offset",
        polar(lambda t: 1.0 + 0.6 * math.cos(t), 32),
        "Two-sided offsets of a dimpled limacon.",
        distances=(0.12, -0.12))
    add("gear_clearance", "cubic_offset",
        polar(lambda t: 1.0 + 0.12 * math.cos(8 * t), 48),
        "Clearance bands about a lobed profile.",
        distances=(0.08, -0.08))
    add("pocketing_stepovers", "cubic_offset", rounded_rectangle(),
        "Exterior stepover rings of a rounded rectangle.",
        distances=(-0.1, -0.2, -0.3))
    add("cam_follower_bands", "cubic_offset",
        polar(lambda t: 1.0 + 0.3 * math.cos(t) + 0.12 * math.cos(2 * t)
              + 0.05 * math.cos(3 * t), 36),
        "Follower clearance bands of a cam lobe.", distances=(0.1, 0.2))
    add("egg_shells", "cubic_offset",
        polar(lambda t: 1.0 / (1.0 + 0.18 * math.cos(t)), 28),
        "Interior and exterior egg shells.", distances=(0.15, -0.15))
    add("squircle_insets", "cubic_offset", superellipse(4.0, 1.4, 1.4),
        "Nested squircle insets.", distances=(0.1, 0.2, 0.3))
    add("peanut_waist_merge", "cubic_offset",
        polar(lambda t: math.sqrt(1.0 + 0.8 * math.cos(2 * t)), 40),
        "Interior offset pinching across the waist.", distances=(0.2,))
    add("trilobe_deep", "cubic_offset",
        polar(lambda t: 1.0 + 0.18 * math.cos(3 * t), 30),
        "Deep cusped offset of a trilobe.", distances=(0.25,))
    add("triangle_min_offsets", "cubic_offset", circle(3, radius=2.0),
        "Offsets of the minimum-size closed cycle.",
        distances=(0.3, -0.3))
    add("dense_circle_rings", "cubic_offset", circle(200),
        "Offsets of a 200-segment source.", distances=(0.5, -0.5))
    add("micro_scale_rings", "cubic_offset", circle(16, radius=1.0e-6),
        "Micron-scale source with proportional offsets.",
        distances=(2.0e-7, -2.0e-7), display_scale=1.0e-6)
    add("far_translated_rings", "cubic_offset",
        circle(12, center=(1.0e9, -1.0e9)),
        "Offsets a billion units from the origin.",
        distances=(0.3, -0.3))
    add("negative_deep_outward", "cubic_offset", circle(24),
        "Large outward offset: area near pi (r + |d|)^2.",
        distances=(-1.5,))
    add("hull_safety_walls", "cubic_offset", convex_hull_points(11),
        "Two-sided safety walls about a hull boundary.",
        distances=(0.15, -0.15))
    add("flat_ellipse_cusp_field", "cubic_offset", ellipse(5.0, 0.25, n=48),
        "Many cusps along a high-aspect source.", distances=(0.15,))
    add("seam_rotated_offset", "cubic_offset",
        circle(16)[5:] + circle(16)[:5],
        "Offset area is independent of the seam location.",
        distances=(0.25,))
    add("cw_star_offsets", "cubic_offset",
        polar(lambda t: 1.0 + 0.35 * math.cos(4 * t), 16)[::-1],
        "Clockwise star with both offset signs.",
        distances=(0.2, -0.2))
    add("gear_beyond_cusps", "cubic_offset",
        polar(lambda t: 1.0 + 0.12 * math.cos(8 * t), 48),
        "Offset far beyond cusp formation.", distances=(0.3,))
    add("stadium_zero_and_inset", "cubic_offset", stadium(),
        "d = 0 alongside a true inset.", distances=(0.0, 0.25))
    add("composite_blob_rings", "cubic_offset",
        catmull_blob(29, keys=10, samples=4),
        "Offset fan of the seeded design blob.",
        distances=(0.12, -0.12))

    # -- 065-096: PH B-spline closed base curves -------------------------
    octagon = [[1.2, 0.0], [0.9, 0.9], [0.0, 1.2], [-0.9, 0.9], [-1.2, 0.0],
               [-0.9, -0.9], [0.0, -1.2], [0.9, -0.9]]
    add("octagon_default_g2", "bspline", octagon,
        "Default G2 continuity, preimage degree 2.")
    add("circle16_g2", "bspline", circle(16),
        "Sixteen-node circle at G2.")
    add("circle16_g3", "bspline", circle(16),
        "G3 request: degree-7 PH spans.", kwargs={"g_order": 3})
    add("circle16_g4", "bspline", circle(16),
        "G4 request: degree-9 PH spans.", kwargs={"g_order": 4})
    add("circle12_c2", "bspline", circle(12),
        "Parametric C2 continuity request.", kwargs={"c_order": 2})
    add("circle12_curvature1", "bspline", circle(12),
        "Arc-length curvature-derivative continuity request.",
        kwargs={"curvature_order": 1})
    add("chord_parameterization", "bspline", octagon,
        "Chord construction parameterization.",
        kwargs={"construction": ConstructionPolicy(parameterization="chord")})
    add("uniform_parameterization", "bspline", octagon,
        "Uniform construction parameterization.",
        kwargs={"construction": ConstructionPolicy(
            parameterization="uniform")})
    add("figure_eight_nu0", "bspline", figure_eight(),
        "Self-intersecting eight: opposite lobes cancel to near zero.",
        fill=False)
    add("limacon_inner_loop", "bspline", limacon(),
        "Limacon with inner loop: the loop counts twice.", fill=False)
    add("trefoil_wave", "bspline",
        polar(lambda t: 1.0 + 0.25 * math.cos(3 * t), 24),
        "Three-lobed simple closed wave.")
    add("star4_g3", "bspline",
        polar(lambda t: 1.0 + 0.35 * math.cos(4 * t), 16),
        "Nonconvex star at G3.", kwargs={"g_order": 3})
    add("flat_ellipse_g2", "bspline", ellipse(4.0, 0.5, n=24),
        "High-aspect closed B-spline.")
    add("egg_g4", "bspline",
        polar(lambda t: 1.0 / (1.0 + 0.18 * math.cos(t)), 20),
        "Egg profile at G4.", kwargs={"g_order": 4})
    add("seeded_blob_g2", "bspline", catmull_blob(41, keys=9, samples=3),
        "Seeded free-form closed blob.")
    add("dimpled_limacon_g3", "bspline",
        polar(lambda t: 1.0 + 0.6 * math.cos(t), 24),
        "Dimpled limacon at G3.", kwargs={"g_order": 3})
    add("kidney_g2", "bspline",
        polar(lambda t: 0.9 + 0.35 * math.cos(t) + 0.18 * math.cos(2 * t), 22),
        "Kidney profile at default continuity.")
    add("cassini_peanut_g2", "bspline",
        polar(lambda t: math.sqrt(1.0 + 0.8 * math.cos(2 * t)), 28),
        "Waisted Cassini oval.")
    add("gear8_g3", "bspline",
        polar(lambda t: 1.0 + 0.12 * math.cos(8 * t), 32),
        "Lobed gear profile at G3.", kwargs={"g_order": 3})
    add("squircle_g4", "bspline", superellipse(4.0, 1.4, 1.4, n=24),
        "Squircle at G4.", kwargs={"g_order": 4})
    add("tiny_scale_1e30", "bspline", circle(12, radius=1.0e-30),
        "Area near 3.1e-60.", display_scale=1.0e-30)
    add("huge_scale_1e30", "bspline", circle(12, radius=1.0e30),
        "Area near 3.1e60.", display_scale=1.0e30)
    add("far_translated_1e6", "bspline", circle(12, center=(1.0e6, 1.0e6)),
        "Unit circle one million units away.")
    add("after_move_edit", "bspline", circle(12),
        "Rendered after one globally repaired move edit (version 1).",
        edit="move")
    add("after_insert_edit", "bspline", circle(12),
        "Rendered after one globally repaired insert edit.", edit="insert")
    add("after_transaction_edit", "bspline", circle(12),
        "Rendered after a globally repaired two-move transaction.",
        edit="transaction")
    add("dense_circle_48", "bspline", circle(48),
        "Ninety-six compiled spans.")
    add("bernoulli_lemniscate", "bspline", bernoulli_lemniscate(),
        "Bernoulli lemniscate: turning number zero.", fill=False)
    add("double_dimple", "bspline",
        polar(lambda t: 1.0 + 0.28 * math.cos(2 * t)
              + 0.2 * math.cos(3 * t), 28),
        "Irregular multi-dimple profile.")
    add("clockwise_octagon", "bspline", octagon[::-1],
        "Clockwise node order: negative signed_area.")
    add("geometric_spacing_g2", "bspline", geometric_circle(),
        "Strongly nonuniform chord spacing.")
    add("configured_max_degree", "bspline", octagon,
        "G3 at the configured preimage-degree cap.",
        kwargs={"g_order": 3,
                "numerics": NumericalPolicy(max_preimage_degree=3)})

    # -- 097-128: PH B-spline closed offsets ------------------------------
    add("octagon_rings", "bspline_offset", octagon,
        "Two-sided rings about the default-G2 octagon.",
        distances=(0.15, -0.15))
    add("circle_g3_shells", "bspline_offset", circle(16),
        "Degree-13 rational offsets of a G3 source.",
        kwargs={"g_order": 3}, distances=(0.2, -0.2))
    add("circle_g4_shell", "bspline_offset", circle(16),
        "Degree-17 rational offset of a G4 source.",
        kwargs={"g_order": 4}, distances=(0.25,))
    add("figure_eight_left", "bspline_offset", figure_eight(),
        "Left offset of a turning-number-zero eight.",
        distances=(0.06,), fill=False)
    add("figure_eight_right", "bspline_offset", figure_eight(),
        "Right offset of the eight: only the -d L0 term flips.",
        distances=(-0.06,), fill=False)
    add("limacon_nu2_offsets", "bspline_offset", limacon(),
        "Turning number two: the pi nu d^2 term doubles.",
        distances=(0.03, -0.03), fill=False)
    add("trefoil_bands", "bspline_offset",
        polar(lambda t: 1.0 + 0.25 * math.cos(3 * t), 24),
        "Two-sided bands about a trefoil wave.",
        distances=(0.1, -0.1))
    add("star4_g3_cusped", "bspline_offset",
        polar(lambda t: 1.0 + 0.35 * math.cos(4 * t), 16),
        "Cusp-forming offset at G3.", kwargs={"g_order": 3},
        distances=(0.3,))
    add("kidney_cusps_g2", "bspline_offset",
        polar(lambda t: 0.9 + 0.35 * math.cos(t) + 0.18 * math.cos(2 * t), 22),
        "Cusped interior offset of the kidney.", distances=(0.25,))
    add("egg_g4_fan", "bspline_offset",
        polar(lambda t: 1.0 / (1.0 + 0.18 * math.cos(t)), 20),
        "Nested interior fan at G4.", kwargs={"g_order": 4},
        distances=(0.12, 0.24))
    add("zero_distance_bitwise", "bspline_offset", octagon,
        "d = 0: bitwise equality with the source signed_area.",
        distances=(0.0,))
    add("gear8_clearance", "bspline_offset",
        polar(lambda t: 1.0 + 0.12 * math.cos(8 * t), 32),
        "Clearance bands about the lobed profile.",
        distances=(0.06, -0.06))
    add("squircle_inset", "bspline_offset", superellipse(4.0, 1.4, 1.4, n=24),
        "Single interior inset of the squircle.", distances=(0.15,))
    add("peanut_waist_pinch", "bspline_offset",
        polar(lambda t: math.sqrt(1.0 + 0.8 * math.cos(2 * t)), 28),
        "Interior offset pinching the waist.", distances=(0.22,))
    add("dimpled_deep_offset", "bspline_offset",
        polar(lambda t: 1.0 + 0.6 * math.cos(t), 24),
        "Deep cusped offset of the dimpled limacon.", distances=(0.3,))
    add("cw_octagon_offsets", "bspline_offset", octagon[::-1],
        "Clockwise source with both offset signs.",
        distances=(0.15, -0.15))
    add("flat_ellipse_cusps", "bspline_offset", ellipse(4.0, 0.5, n=24),
        "Cusp field along a high-aspect source.", distances=(0.1,))
    add("far_translated_shells", "bspline_offset",
        circle(12, center=(1.0e6, 1.0e6)),
        "Shells one million units from the origin.",
        distances=(0.2, -0.2))
    add("micro_scale_shells", "bspline_offset", circle(12, radius=1.0e-6),
        "Micron-scale shells.", distances=(2.0e-7, -2.0e-7),
        display_scale=1.0e-6)
    add("edited_source_offsets", "bspline_offset", circle(12),
        "Offsets taken after a globally repaired move edit.",
        edit="move", distances=(0.15, -0.15))
    add("transaction_source_offsets", "bspline_offset", circle(12),
        "Offsets after a globally repaired two-move transaction.",
        edit="transaction", distances=(0.2,))
    add("blob_rings", "bspline_offset", catmull_blob(41, keys=9, samples=3),
        "Ring fan about the seeded blob.", distances=(0.12, -0.12))
    add("reversed_loop_deep", "bspline_offset", circle(12),
        "Offset beyond every curvature radius: reversed loop.",
        distances=(1.6,))
    add("negative_outward_deep", "bspline_offset", circle(12),
        "Large outward offset.", distances=(-1.0,))
    add("dense48_rings", "bspline_offset", circle(48),
        "Rings about a ninety-six-span source.", distances=(0.3,))
    add("lemniscate_offsets", "bspline_offset", bernoulli_lemniscate(),
        "Offsets of the Bernoulli lemniscate (nu = 0).",
        distances=(0.05,), fill=False)
    add("double_dimple_offsets", "bspline_offset",
        polar(lambda t: 1.0 + 0.28 * math.cos(2 * t)
              + 0.2 * math.cos(3 * t), 28),
        "Cusped offset of the multi-dimple profile.", distances=(0.15,))
    add("c2_shells", "bspline_offset", circle(12),
        "Offsets of a parametric-C2 source.", kwargs={"c_order": 2},
        distances=(0.2,))
    add("curvature1_shells", "bspline_offset", circle(12),
        "Offsets of a curvature-order source.",
        kwargs={"curvature_order": 1}, distances=(0.2,))
    add("chord_param_offsets", "bspline_offset", octagon,
        "Offsets under chord parameterization.",
        kwargs={"construction": ConstructionPolicy(parameterization="chord")},
        distances=(0.2,))
    add("uniform_param_offsets", "bspline_offset", octagon,
        "Offsets under uniform parameterization.",
        kwargs={"construction": ConstructionPolicy(
            parameterization="uniform")},
        distances=(0.2,))
    add("configured_max_offsets", "bspline_offset", octagon,
        "Offsets at the configured preimage-degree cap.",
        kwargs={"g_order": 3,
                "numerics": NumericalPolicy(max_preimage_degree=3)},
        distances=(0.2,))

    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                        help="verify all cases without rendering")
    parser.add_argument("--only", type=int, default=None,
                        help="run a single 1-based case index")
    parser.add_argument("--start", type=int, default=1,
                        help="first 1-based case index to run")
    parser.add_argument("--end", type=int, default=128,
                        help="last 1-based case index to run")
    args = parser.parse_args()
    cases = all_cases()
    if len(cases) != 128:
        raise SystemExit(f"expected exactly 128 cases, found {len(cases)}")
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        duplicated = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(f"case names must be unique: {duplicated}")
    failures: list[str] = []
    for index, case in enumerate(cases, start=1):
        if args.only is not None and index != args.only:
            continue
        if not args.start <= index <= args.end:
            continue
        try:
            render(index, case, args.check_only)
        except Exception as error:  # noqa: BLE001 - report and continue
            failures.append(
                f"{index:03d}_{case.name}: {type(error).__name__}: {error}"
            )
            continue
        print(f"{'verified' if args.check_only else 'rendered'} "
              f"{index:03d}_{case.name}")
    if failures:
        print("\nFAILURES:")
        for line in failures:
            print(" ", line)
        raise SystemExit(1)
    print(f"\nAll cases {'verified' if args.check_only else 'rendered'} "
          f"into {OUT}")


if __name__ == "__main__":
    main()
