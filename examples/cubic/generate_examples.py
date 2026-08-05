"""Generate 64 example splines - ordinary, organic, and extreme - as PNGs.

Run:  python examples/cubic/generate_examples.py
Output: examples/cubic/base/NN_name.png

Every example is fully constructed by ``CubicPHSpline`` and passes an
independent invariant spot-check before it is drawn.
"""

from __future__ import annotations

import math
import os

import numpy as np
from _common import (
    circle_points,
    graph_points,
    log_spiral_points,
    plot_example,
    polyline_from_turns,
)
from scipy.spatial import ConvexHull

OUT = os.path.join(os.path.dirname(__file__), "base")


def random_convex_arc(seed: int, n_cloud: int, keep: int) -> list:
    """Open arc of convex-hull vertices of a seeded random cloud."""
    rng = np.random.default_rng(seed)
    cloud = rng.normal(size=(n_cloud, 2)) * [2.0, 1.3]
    hull = ConvexHull(cloud)
    verts = cloud[hull.vertices]  # counterclockwise order
    k = min(keep, len(verts) - 1)
    return [[float(x), float(y)] for x, y in verts[:k]]


def superellipse_arc(a, b, p, t0, t1, n):
    ts = np.linspace(t0, t1, n)
    return [
        [
            float(a * math.copysign(abs(math.cos(t)) ** (2.0 / p), math.cos(t))),
            float(b * math.copysign(abs(math.sin(t)) ** (2.0 / p), math.sin(t))),
        ]
        for t in ts
    ]


EXAMPLES: list[tuple[str, list, str]] = []


def add(name: str, points: list, note: str = "") -> None:
    EXAMPLES.append((name, points, note))


# ---------------------------------------------------------------------------
# Ordinary (1-20)
# ---------------------------------------------------------------------------

add("two_point_segment", [[0.0, 0.0], [3.0, 4.0]], "degenerate straight spline")
add("collinear_uniform", [[float(i), 0.5 * i] for i in range(5)])
add(
    "collinear_nonuniform",
    [[x, -0.25 * x] for x in [0.0, 0.13, 0.5, 1.2, 2.9, 5.5, 9.1, 13.0]],
)
add("gentle_arc_3pt", circle_points(R=4.0, a0=1.2, a1=1.75, n=3), "minimal curved case")
add("quarter_circle", circle_points(R=2.0, a0=0.0, a1=math.pi / 2, n=6))
add("half_circle", circle_points(R=1.5, a0=-0.2, a1=-0.2 + math.pi, n=12))
add("horseshoe_240deg", circle_points(R=1.0, a0=-0.6, a1=-0.6 + 4.19, n=16))
add("dense_circle_arc", circle_points(R=3.0, a0=0.4, a1=2.6, n=40))
add("clockwise_quarter", circle_points(R=2.5, a0=0.1, a1=1.67, n=9, cw=True))
add("parabola_bowl", graph_points(lambda x: 0.6 * x * x, -1.8, 1.8, 9))
add("exponential_growth", graph_points(math.exp, -1.5, 1.6, 8))
add("logarithm_curve", graph_points(math.log, 0.25, 4.0, 8))
add("square_root_curve", graph_points(math.sqrt, 0.05, 4.0, 8))
add("cubic_branch", graph_points(lambda x: x**3, 0.25, 1.6, 7))
add("hyperbola_branch", graph_points(lambda x: 1.0 / x, 0.35, 3.0, 7))
add("catenary", graph_points(math.cosh, -1.4, 1.4, 9))
add("sine_arch", graph_points(math.sin, 0.15, math.pi - 0.15, 9))
add("power_three_halves", graph_points(lambda x: x**1.5, 0.15, 2.4, 8))
add(
    "ellipse_arc",
    [[2.5 * math.cos(t), 1.2 * math.sin(t)] for t in np.linspace(-1.1, 1.3, 10)],
)
add(
    "wide_shallow_arc",
    circle_points(R=50.0, a0=1.51, a1=1.63, n=5),
    "large radius, small arc",
)

# ---------------------------------------------------------------------------
# Organic (21-44)
# ---------------------------------------------------------------------------

add("golden_spiral", log_spiral_points(a=0.6, b=0.3063, t0=0.0, t1=3.6 * math.pi, n=18))
add("nautilus", log_spiral_points(a=0.4, b=0.18, t0=0.0, t1=4.4 * math.pi, n=28))
add(
    "archimedean_coil",
    [
        [(0.4 + 0.22 * t) * math.cos(t), (0.4 + 0.22 * t) * math.sin(t)]
        for t in np.linspace(0.0, 4.6 * math.pi, 24)
    ],
)
add(
    "fern_curl",
    log_spiral_points(a=1.0, b=-0.11, t0=0.0, t1=7 * math.pi, n=36),
    "inward log spiral",
)
add("snail_shell", log_spiral_points(a=0.25, b=0.14, t0=0.5, t1=5.6 * math.pi, n=30))
add("vine_tendril", log_spiral_points(a=0.8, b=0.05, t0=0.0, t1=6 * math.pi, n=40))
add(
    "seashell_whorl",
    log_spiral_points(a=0.5, b=0.21, t0=1.0, t1=4 * math.pi, n=22, cw=True),
)
add("egg_profile", superellipse_arc(1.0, 1.35, 2.6, 0.12, math.pi - 0.12, 12))
add(
    "leaf_edge",
    [[t, 1.7 * math.sin(t) * math.exp(-0.25 * t)] for t in np.linspace(0.25, 2.6, 10)],
    "damped-sine leaf margin (convex section)",
)
add("petal_curve", superellipse_arc(1.6, 0.9, 1.7, 0.25, 2.6, 9))
add(
    "raindrop_flank",
    [
        [math.sin(t) ** 1.35, -math.cos(t)]
        for t in np.linspace(0.18, math.pi - 0.75, 11)
    ],
)
add("dune_crest", graph_points(lambda x: 1.1 * math.sin(x) ** 0.8, 0.35, 2.6, 8))
add(
    "wave_swell",
    circle_points(R=1.8, a0=2.62, a1=0.62, n=10, cw=True, center=(0.0, 1.5)),
)
add(
    "river_meander_bend",
    circle_points(R=6.0, a0=-1.1, a1=1.05, n=9, center=(0.0, -4.0)),
)
add(
    "bird_wing_leading_edge",
    [
        [0.0, 0.0],
        [0.9, 0.62],
        [1.9, 1.05],
        [3.0, 1.3],
        [4.1, 1.38],
        [5.2, 1.32],
        [6.2, 1.12],
        [7.1, 0.78],
        [7.8, 0.35],
        [8.3, -0.15],
    ],
    "hand-set freeform convex profile",
)
add(
    "moon_crescent_outer",
    circle_points(R=1.0, a0=0.35, a1=2 * math.pi - 0.35, n=14, cw=True),
)
add(
    "fiddlehead",
    log_spiral_points(a=1.4, b=-0.075, t0=0.0, t1=9 * math.pi, n=48),
    "4.5-turn curl",
)
add("pumpkin_rib", superellipse_arc(1.1, 1.5, 3.2, 0.3, math.pi - 0.3, 9))
add(
    "clam_shell_edge",
    circle_points(R=2.2, a0=math.pi + 0.4, a1=2 * math.pi - 0.4, n=12),
)
add(
    "rose_petal_edge",
    [
        [1.3 * math.sin(t) ** 0.7 * math.cos(t - 0.5), 1.5 * math.sin(t) * 0.9]
        for t in np.linspace(0.35, 1.85, 11)
    ],
    "sculpted petal rim",
)
add(
    "random_convex_hull_a",
    random_convex_arc(seed=11, n_cloud=260, keep=15),
    "seeded random hull arc",
)
add(
    "random_convex_hull_b",
    random_convex_arc(seed=97, n_cloud=900, keep=24),
    "seeded random hull arc",
)
add("heart_lobe", circle_points(R=1.0, a0=-0.9, a1=1.9, n=10, center=(0.55, 0.9)))
add(
    "spiral_galaxy_arm",
    log_spiral_points(a=0.3, b=0.24, t0=0.8, t1=3.4 * math.pi, n=26),
)

# ---------------------------------------------------------------------------
# Extreme (45-64)
# ---------------------------------------------------------------------------

add(
    "near_reversal_turn",
    polyline_from_turns([1.0, 1.0, 1.0], [0.995 * math.pi, 0.4]),
    f"single turn = 0.995 pi = {0.995 * math.pi:.5f} rad",
)
add(
    "pair_sum_high",
    polyline_from_turns([1.0, 1.0, 1.0], [2.04, 2.04]),
    "adjacent turn sum 4.08 of allowed 4.09691 (99.6%)",
)
add(
    "chord_ratio_1e6",
    polyline_from_turns(list(10.0 ** np.linspace(-6, 0, 8)), [0.3] * 7),
    "chord lengths span 6 decades",
)
add(
    "chord_ratio_1e9",
    polyline_from_turns(list(10.0 ** np.linspace(-9, 0, 10)), [0.25] * 9),
    "chord lengths span 9 decades",
)
add(
    "tiny_curvature_1e-9",
    circle_points(R=1e9, a0=0.7, a1=0.7 + 1.2e-3, n=6),
    "circle radius 1e9: |kappa| = 1e-9",
)
add(
    "high_curvature_1e6",
    circle_points(R=1e-6, a0=0.2, a1=1.9, n=9),
    "circle radius 1e-6: |kappa| = 1e6",
)
add(
    "coords_1e120",
    [[1e120 * p[0], 1e120 * p[1]] for p in circle_points(R=1.3, a0=0.1, a1=1.8, n=9)],
    "coordinates of magnitude 1e120",
)
add(
    "coords_1e-120",
    [[1e-120 * p[0], 1e-120 * p[1]] for p in circle_points(R=1.3, a0=0.1, a1=1.8, n=9)],
    "coordinates of magnitude 1e-120",
)
add(
    "far_offset_1e12",
    circle_points(R=250.0, a0=0.3, a1=1.9, n=9, center=(1e12, -1e12)),
    "small arc a trillion units from the origin",
)
add(
    "longest_line",
    [[-4e307, -3e307], [4e307, 3e307]],
    "straight span of 1e308 units, near the binary64 limit",
)
add(
    "collinear_1000pts",
    [[float(i) + 0.3 * math.sin(0.0), float(i) * 0.25] for i in range(1000)],
    "1000 collinear points",
)
add(
    "spiral_400pts",
    log_spiral_points(a=0.5, b=0.07, t0=0.0, t1=7 * math.pi, n=400),
    "399 segments through the sparse tridiagonal solver",
)
add(
    "total_turn_6pi",
    log_spiral_points(a=1.0, b=0.05, t0=0.0, t1=6 * math.pi, n=80),
    "total tangent turn of ~6 pi (three full revolutions)",
)
add(
    "clockwise_golden_spiral",
    log_spiral_points(a=0.6, b=0.3063, t0=0.0, t1=3 * math.pi, n=20, cw=True),
)
add(
    "ellipse_aspect_1000",
    [
        [1000.0 * math.cos(t), math.sin(t)]
        for t in np.linspace(0.15, math.pi - 0.15, 24)
    ],
    "ellipse arc with 1000:1 aspect ratio",
)
add(
    "geometric_spacing_2x",
    polyline_from_turns([2.0 ** (-k) for k in range(20)], [0.18] * 19),
    "every chord half the previous (ratio 2^19)",
)
add(
    "single_turn_3rad",
    polyline_from_turns([1.0, 1.0, 1.0], [3.0, 0.35]),
    "one 3.0 rad turn (171.9 deg)",
)
add(
    "loopy_speed_dip",
    polyline_from_turns([1.0, 0.3, 0.35], [1.7, 1.7]),
    "short middle chord with 1.7 rad turns on both ends (deep speed dip)",
)
add(
    "boundary_clamp_case",
    polyline_from_turns([1.0, 1.0, 1.0, 1.0], [2.9, 0.4, 0.4]),
    "start circumcircle tangent clamped by the uniqueness bound",
)
add(
    "grand_composite",
    [
        [
            (0.4 * math.exp(0.09 * t) * (1.0 + 0.0 * t)) * math.cos(t),
            (0.4 * math.exp(0.09 * t)) * math.sin(t),
        ]
        for t in np.concatenate(
            [
                np.linspace(0.0, 2 * math.pi, 12, endpoint=False),
                np.linspace(2 * math.pi, 4 * math.pi, 28, endpoint=False),
                np.linspace(4 * math.pi, 6.5 * math.pi, 80),
            ]
        )
    ],
    "spiral with three different sampling densities",
)


def main() -> None:
    assert len(EXAMPLES) == 64, f"expected 64 examples, have {len(EXAMPLES)}"
    failures = []
    for i, (name, pts, note) in enumerate(EXAMPLES, start=1):
        try:
            curve = plot_example(i, name, pts, OUT, note=note)
            print(f"ok  {i:02d} {name:28s} {curve!r}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((i, name, exc))
            print(f"FAIL {i:02d} {name:28s} {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit(f"{len(failures)} example(s) failed")
    print(f"\nAll 64 examples rendered to {OUT}")


if __name__ == "__main__":
    main()
