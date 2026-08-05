"""64 direct general-data examples: letters, tracks, maps,
topological constructs, illustrations, and mechanical objects.

Run:  python examples/cubic/generate_nonconvex_shapes.py
Output: examples/cubic/nonconvex/NN_name.png

Every dataset is passed directly to one ``CubicPHSpline``.  Each plot marks
auxiliary inflection points with translucent magenta crosses and 10
arc-equidistant points with red circles, then independently checks the
red-point spacing against a dense chordal measurement of the rendered geometry.
"""

from __future__ import annotations

import math
import os

import numpy as np
from _composite import Turtle, arc_pts, param, plot_nonconvex

OUT = os.path.join(os.path.dirname(__file__), "nonconvex")

EXAMPLES: list[tuple[str, list, str]] = []


def add(name: str, pts, note: str = "") -> None:
    EXAMPLES.append((name, pts, note))


def add_turtle(name: str, t: Turtle, note: str = "") -> None:
    add(name, t.data(), note)


def with_end_support(points, *, start: bool = True, end: bool = True):
    """Split terminal spans to make an intended straight tangent explicit."""
    out = [list(map(float, p)) for p in points]
    if len(out) < 2:
        return out
    if start:
        a, b = out[0], out[1]
        out.insert(1, [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])])
    if end:
        a, b = out[-2], out[-1]
        out.insert(-1, [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])])
    return out


def closed_catmull_rom(keys, samples_per_span: int = 8) -> list[list[float]]:
    """Sample a periodic Catmull-Rom outline but stop just short of closure."""
    a = np.asarray(keys, dtype=np.float64)
    out: list[list[float]] = []
    count = a.shape[0]

    def at(i: int, t: float) -> np.ndarray:
        p0, p1 = a[(i - 1) % count], a[i]
        p2, p3 = a[(i + 1) % count], a[(i + 2) % count]
        return 0.5 * (
            2.0 * p1
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
        )

    for i in range(count):
        for k in range(samples_per_span):
            out.append(at(i, k / samples_per_span).tolist())
    # The spline class is open.  This final point makes a visually closed
    # outline without duplicating the first point.
    out.append(at(count - 1, 0.985).tolist())
    return out


# ---------------------------------------------------------------------------
# Letters and glyphs (1-10)
# ---------------------------------------------------------------------------

# S: two internally tangent circle arcs; the tangency makes the inflection
# smooth by construction.
_s = arc_pts((0.5, 0.85), 0.28, -0.35, 1.5 * math.pi, 22)
_s += arc_pts((0.5, 0.29), 0.28, 0.5 * math.pi, -2.24, 24)[1:]
add("letter_S", _s, "two tangent arcs, one inflection")

add(
    "letter_Z",
    with_end_support([[0.0, 1.2], [0.9, 1.2], [0.0, 0.0], [0.9, 0.0]]),
    "Z stroke with supported straight terminals",
)

add(
    "letter_N",
    with_end_support([[0.0, 0.0], [0.2, 1.3], [0.9, 0.0], [1.1, 1.3]]),
    "open N stroke with supported straight terminals",
)

add(
    "letter_W",
    param(
        lambda t: t, lambda t: 0.65 + 0.58 * math.cos(4.0 * math.pi * t), 0.0, 1.0, 41
    ),
    "smooth open W stroke",
)

_2 = arc_pts((0.5, 0.95), 0.34, 2.6, -0.35, 22)
_2 += [[0.14, 0.16], [0.82, 0.16]]
add(
    "letter_2",
    with_end_support(_2, start=False),
    "arc bowl into diagonal and supported straight base bar",
)

_3 = arc_pts((0.42, 0.98), 0.30, 2.35, -1.35, 22)
_3 += arc_pts((0.42, 0.36), 0.32, 1.30, -2.45, 24)[1:]
add("letter_3", _3, "two bowls meeting at a concave notch")

_5 = [[0.82, 1.30], [0.20, 1.30], [0.20, 1.05], [0.23, 0.79]]
_5 += arc_pts((0.48, 0.39), 0.35, 2.15, -2.75, 28)
add(
    "letter_5",
    with_end_support(_5, start=True, end=False),
    "supported top bar, stem, shoulder and complete lower bowl",
)

_t = Turtle(pos=(0.15, 1.30), heading=0.0)
_t.straight(0.45, 2).arc(0.14, -0.5 * math.pi, 6)
_t.straight(0.68, 4).arc(0.29, -2.45, 12)
add_turtle("letter_J", _t, "straight top and stem joined by a fillet into the hook")

_q = arc_pts((0.5, 0.98), 0.32, math.pi + 0.4, -0.5 * math.pi, 24)
_q += [[0.5, 0.6 - t] for t in np.linspace(0.05, 0.42, 4)]
add("letter_question", _q, "question-mark bowl and stem")

_n8 = 130
add(
    "glyph_eight",
    param(
        lambda t: 0.8 * math.sin(t) * math.cos(t) / (1.0 + math.sin(t) ** 2),
        lambda t: 1.2 * math.cos(t) / (1.0 + math.sin(t) ** 2),
        0.06,
        2.0 * math.pi - 0.06,
        _n8,
    ),
    "upright Bernoulli-style figure-eight glyph",
)

# ---------------------------------------------------------------------------
# Racing tracks (11-18)
# ---------------------------------------------------------------------------

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
_t.straight(3.0, 6).arc(0.55, math.pi * 0.9).straight(0.6, 2)
_t.arc(0.35, -1.5).arc(0.35, 1.5).straight(1.6, 3)
_t.arc(0.5, math.pi * 0.85).straight(0.7, 2).arc(0.9, 1.1)
_t.straight(1.0, 3).arc(0.45, math.pi * 0.75).straight(0.7, 2)
add_turtle("gp_circuit", _t, "straights, chicane, hairpins, sweepers")

_t = Turtle(pos=(0.0, 0.0), heading=0.25)
for k in range(5):
    sgn = 1.0 if k % 2 == 0 else -1.0
    _t.straight(1.5 - 0.12 * k, 3).arc(0.28, sgn * (math.pi - 0.28))
_t.straight(1.2, 3)
add_turtle("mountain_switchbacks", _t, "five alternating hairpins")

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
_t.straight(2.6, 5).arc(0.8, math.pi).straight(0.9, 2)
_t.arc(0.3, -1.25).arc(0.3, 1.25).straight(0.9, 2).arc(0.8, math.pi)
add_turtle("oval_with_chicane", _t, "speedway oval broken by one chicane")

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
_t.straight(0.9, 2).arc(0.30, 2.4).arc(0.5, -1.1).arc(0.22, 2.9)
_t.straight(0.5, 2).arc(0.35, -2.6).arc(0.28, 1.9).straight(0.4, 2)
_t.arc(0.45, 2.2).arc(0.3, -2.0).straight(0.7, 2)
add_turtle("kart_track", _t, "tight mixed-direction kart circuit")

_t = Turtle(pos=(0.0, 0.0), heading=0.5)
_t.arc(1.0, 2.4).straight(1.4, 3).arc(0.55, -4.4).straight(1.4, 3).arc(1.0, 2.0)
add_turtle("figure_eight_speedway", _t, "crossover circuit (self-crossing)")

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for L, r, a in [
    (1.6, 0.12, math.pi / 2),
    (0.8, 0.12, math.pi / 2),
    (0.7, 0.12, -math.pi / 2),
    (1.0, 0.12, -math.pi / 2),
    (0.6, 0.12, math.pi / 2),
    (1.1, 0.12, math.pi / 2),
    (0.5, 0.12, -math.pi / 2),
]:
    _t.straight(L, 3).arc(r, a)
_t.straight(0.8, 2)
add_turtle("street_circuit", _t, "city block right-angle corners, filleted")

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
_t.straight(4.0, 8).arc(0.35, math.pi - 0.15).straight(3.7, 8).arc(0.5, 2.2)
add_turtle("drag_strip_return", _t, "long straight, tight turnaround loop")

rng = np.random.default_rng(42)
_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for _ in range(14):
    _t.straight(0.35 + 0.9 * float(rng.random()), 2)
    _t.arc(
        0.25 + 0.6 * float(rng.random()),
        float(rng.uniform(0.7, 2.4)) * (1 if rng.random() < 0.5 else -1),
    )
add_turtle("endurance_circuit", _t, "seeded 14-corner endurance layout")

# ---------------------------------------------------------------------------
# Maps (19-26)
# ---------------------------------------------------------------------------

rng = np.random.default_rng(7)
_t = Turtle(pos=(0.0, 0.0), heading=0.3)
for k in range(9):
    sgn = 1.0 if k % 2 == 0 else -1.0
    _t.arc(0.45 + 0.5 * float(rng.random()), sgn * float(rng.uniform(1.9, 2.7)))
add_turtle("river_meanders", _t, "alternating oxbow bends (seeded)")

rng = np.random.default_rng(19)
_t = Turtle(pos=(0.0, 0.0), heading=0.1)
for _ in range(26):
    _t.arc(
        0.14 + 0.5 * float(rng.random()),
        float(rng.uniform(0.5, 1.6)) * (1 if rng.random() < 0.5 else -1),
    )
add_turtle("coastline", _t, "random-walk coastal trace (seeded)")

_t = Turtle(pos=(0.0, 0.0), heading=0.15)
for depth in (1.0, 1.6, 0.8, 1.3):
    _t.straight(0.5, 2).arc(0.18, math.pi - 0.35)
    _t.straight(depth, 3).arc(0.14, -(math.pi - 0.25))
    _t.straight(depth * 0.9, 3).arc(0.18, math.pi - 0.35)
    _t.straight(0.4, 2).arc(0.35, -1.2)
add_turtle("fjord_coast", _t, "deep narrow inlets with near-reversal mouths")

add(
    "island_outline",
    param(
        lambda t: (
            (1.0 + 0.23 * math.cos(3 * t + 0.8) + 0.11 * math.cos(5 * t)) * math.cos(t)
        ),
        lambda t: (
            (1.0 + 0.23 * math.cos(3 * t + 0.8) + 0.11 * math.cos(5 * t)) * math.sin(t)
        ),
        0.05,
        2 * math.pi - 0.05,
        140,
    ),
    "Fourier blob with bays and headlands",
)

_t = Turtle(pos=(0.0, 0.0), heading=1.1)
for k in range(6):
    sgn = 1.0 if k % 2 == 0 else -1.0
    _t.straight(0.55 + 0.1 * k, 2).arc(0.16, sgn * (math.pi - 0.5))
_t.straight(0.9, 3)
add_turtle("alpine_road", _t, "hairpin switchbacks climbing a slope")

rng = np.random.default_rng(23)
_p = [[0.0, 0.0]]
for k in range(40):
    _p.append(
        [
            _p[-1][0] + 0.25 + 0.15 * float(rng.random()),
            0.9 * math.sin(0.55 * k + 0.9 * float(rng.random()))
            + 0.35 * float(rng.random()),
        ]
    )
add("mountain_ridge", _p, "jagged seeded ridge profile")

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
_t.straight(1.4, 3).arc(0.5, 1.9).straight(0.5, 2).arc(0.35, -2.4)
_t.straight(0.8, 2).arc(0.6, 2.1).arc(0.4, -1.7).straight(1.1, 3)
add_turtle("harbor_breakwater", _t, "breakwater walls and turning basins")

_t = Turtle(pos=(0.0, 0.0), heading=0.6)
for _ in range(4):
    _t.arc(0.8, 1.4).arc(0.16, -(math.pi - 0.4)).arc(0.8, 1.4)
add_turtle("dune_field", _t, "slip-face dune crests with sharp brinks")

# ---------------------------------------------------------------------------
# Topological constructs (27-38)
# ---------------------------------------------------------------------------

_n = 180
add(
    "trefoil_projection",
    param(
        lambda t: (1.0 + 0.30 * math.cos(3.0 * t)) * math.cos(t),
        lambda t: (1.0 + 0.30 * math.cos(3.0 * t)) * math.sin(t),
        0.04,
        2 * math.pi - 0.04,
        _n,
    ),
    "open three-lobe trefoil-inspired outline",
)
_n = 220
add(
    "torus_knot_2_3",
    param(
        lambda t: t + 0.18 * math.sin(3.0 * t),
        lambda t: 0.65 * math.sin(2.0 * t),
        0.0,
        2 * math.pi,
        _n,
    ),
    "open 2:3 braided-wave projection",
)
_n = 130
add(
    "bernoulli_lemniscate",
    param(
        lambda t: 1.3 * math.cos(t) / (1.0 + math.sin(t) ** 2),
        lambda t: 1.3 * math.sin(t) * math.cos(t) / (1.0 + math.sin(t) ** 2),
        0.06,
        2 * math.pi - 0.06,
        _n,
    ),
    "lemniscate of Bernoulli",
)
_n = 170
add(
    "lissajous_2_3",
    param(
        lambda t: math.sin(2 * t + 0.9),
        lambda t: math.sin(3 * t),
        0.0,
        2 * math.pi - 0.002,
        _n,
    ),
    "near-closed open Lissajous 2:3",
)
_n = 230
add(
    "lissajous_3_4",
    param(
        lambda t: (1.0 + 0.12 * math.cos(3 * t) + 0.06 * math.cos(4 * t)) * math.cos(t),
        lambda t: (1.0 + 0.12 * math.cos(3 * t) + 0.06 * math.cos(4 * t)) * math.sin(t),
        0.0,
        2 * math.pi - 0.002,
        _n,
    ),
    "near-closed open 3:4 harmonic loop",
)
_n = 300
add(
    "epitrochoid_5_3",
    param(
        lambda t: (1.0 + 0.22 * math.cos(5.0 * t)) * math.cos(t),
        lambda t: (1.0 + 0.22 * math.cos(5.0 * t)) * math.sin(t),
        0.03,
        2 * math.pi - 0.03,
        _n,
    ),
    "five-lobe open epitrochoid-style rim",
)
_n = 320
add(
    "hypotrochoid_7_3",
    param(
        lambda t: (0.8 + 0.28 * math.cos(7.0 * t)) * math.cos(t),
        lambda t: (0.8 + 0.28 * math.cos(7.0 * t)) * math.sin(t),
        0.03,
        2 * math.pi - 0.03,
        _n,
    ),
    "seven-lobe open hypotrochoid-style rim",
)
_n, _t0, _t1 = 200, 0.25, 6 * math.pi - 0.25
add(
    "cycloid_arches",
    param(lambda t: t - math.sin(t), lambda t: 1.0 - math.cos(t), _t0, _t1, _n),
    "three open cycloid arches",
)
_n, _t0, _t1 = 140, 0.12, 2 * math.pi - 0.12
add(
    "cardioid",
    param(
        lambda t: (1.0 + math.cos(t)) * math.cos(t),
        lambda t: (1.0 + math.cos(t)) * math.sin(t),
        _t0,
        _t1,
        _n,
    ),
    "open cardioid sampled away from its cusp",
)
_n, _t0, _t1 = 170, 0.10, 2 * math.pi - 0.10
add(
    "deltoid",
    param(
        lambda t: 2.0 * math.cos(t) + math.cos(2.0 * t),
        lambda t: 2.0 * math.sin(t) - math.sin(2.0 * t),
        _t0,
        _t1,
        _n,
    ),
    "open deltoid sampled around its three lobes",
)
_n = 260
add(
    "rose_4_petals",
    param(
        lambda t: (1.0 + 0.38 * math.cos(4.0 * t)) * math.cos(t),
        lambda t: (1.0 + 0.38 * math.cos(4.0 * t)) * math.sin(t),
        0.04,
        2 * math.pi - 0.04,
        _n,
    ),
    "four-petal open rose outline without a repeated center point",
)
_kappa_s = lambda s: 2.2 * math.sin(1.5 * s)
_h = 0.0
_pos = [0.0, 0.0]
_euler = [[0.0, 0.0]]
for _k in range(240):
    _h += _kappa_s(_k * 0.05) * 0.05
    _pos = [_pos[0] + 0.05 * math.cos(_h), _pos[1] + 0.05 * math.sin(_h)]
    _euler.append(list(_pos))
add(
    "euler_serpentine",
    _euler,
    "curvature prescribed as a sine of arc length (clothoid-like S chain)",
)

# ---------------------------------------------------------------------------
# Illustrations (39-51)
# ---------------------------------------------------------------------------

add(
    "heart",
    param(
        lambda t: (16.0 * math.sin(t) ** 3) / 17.0,
        lambda t: (
            (
                13.0 * math.cos(t)
                - 5.0 * math.cos(2 * t)
                - 2.0 * math.cos(3 * t)
                - math.cos(4 * t)
            )
            / 17.0
        ),
        0.09,
        2 * math.pi - 0.09,
        150,
    ),
    "heart outline: concave notch and bottom point",
)

_cr = arc_pts((0.0, 0.0), 1.0, -1.22, 1.22, 40)
_cr += arc_pts((0.34, 0.0), 0.936, 1.47, -1.47, 36)[1:]
add("crescent_moon", _cr, "outer and inner limbs meeting at a sharp horn")

_t = Turtle(pos=(0.0, 0.0), heading=0.9)
for r in (0.42, 0.55, 0.38, 0.5, 0.33):
    _t.arc(r, 3.1).arc(0.3 * r, -1.55)
add_turtle("cloud_outline", _t, "puffy lobes with concave junctions")

_t = Turtle(pos=(-1.0, 0.0), heading=1.25)
_t.arc(1.05, -2.5)
_t.corner(-1.34)
for k in range(4):
    _t.arc(0.48, -1.1)
    if k < 3:
        _t.corner(1.1)
add_turtle("umbrella_canopy", _t, "canopy dome over four scallops")

_t = Turtle(pos=(0.0, 0.0), heading=1.85)
_t.arc(0.55, -2.1).arc(0.14, 2.6).arc(0.5, -2.5).arc(0.14, 2.6).arc(0.55, -2.1)
add_turtle("tulip_bloom", _t, "three petals with sharp valleys")

_fish = closed_catmull_rom(
    [
        (1.00, 0.23),
        (0.55, 0.52),
        (0.00, 0.58),
        (-0.70, 0.45),
        (-1.05, 0.00),
        (-0.70, -0.45),
        (0.00, -0.58),
        (0.55, -0.52),
        (1.00, -0.23),
        (1.18, -0.28),
        (1.58, -0.62),
        (1.38, 0.00),
        (1.58, 0.62),
        (1.18, 0.28),
    ]
)
add("fish_outline", _fish, "near-closed smooth body with a complete forked tail")

_t = Turtle(pos=(0.0, 0.0), heading=0.9)
_t.arc(0.65, -1.8).arc(0.28, 1.9).arc(0.65, -1.8)
add_turtle("bird_glide", _t, "gliding-bird double arch")

add(
    "snake_slither",
    param(
        lambda t: t,
        lambda t: 0.62 * math.exp(-0.13 * t) * math.sin(2.1 * t),
        0.0,
        10.0,
        150,
    ),
    "damped slither: turns decay smoothly toward straight",
)

_t = Turtle(pos=(0.0, 0.0), heading=math.pi / 2)
_t.arc(0.9, -1.35).arc(0.35, 1.75).straight(0.55, 2).arc(0.28, 1.35).arc(1.1, -0.8)
add_turtle("wine_glass_profile", _t, "bowl, stem and foot in one stroke")

_t = Turtle(pos=(0.0, 0.0), heading=math.pi / 2)
_t.arc(0.62, 2.5).arc(0.30, -1.75).arc(0.80, 2.6)
add_turtle("guitar_body_half", _t, "upper bout, waist, lower bout")

_t = Turtle(pos=(0.0, 0.0), heading=math.pi / 2)
_t.arc(0.5, -1.3).arc(0.42, 1.6).arc(0.30, -1.5).arc(0.55, 1.15)
add_turtle("vase_profile", _t, "classic S-curved silhouette")

add(
    "candle_flame",
    param(
        lambda t: 0.42 * math.sin(t) * (1.0 + 0.5 * math.cos(t)),
        lambda t: -0.85 * math.cos(t) + 0.18 * math.cos(2.0 * t),
        0.16,
        2 * math.pi - 0.16,
        110,
    ),
    "teardrop flame profile",
)

_t = Turtle(pos=(-0.9, 0.0), heading=1.1)
_t.arc(1.0, -2.2)
_t.corner(-1.4)
for k in range(5):
    _t.arc(0.20, -1.35)
    if k < 4:
        _t.corner(1.35)
add_turtle("jellyfish_dome", _t, "dome with five trailing frill scallops")

# ---------------------------------------------------------------------------
# Mechanical / technical (52-64)
# ---------------------------------------------------------------------------


def _smooth_square(s: float, sharp: float = 0.05) -> float:
    return s / math.sqrt(s * s + sharp)


add(
    "gear_wheel_profile",
    param(
        lambda t: (1.0 + 0.13 * _smooth_square(math.sin(9.0 * t))) * math.cos(t),
        lambda t: (1.0 + 0.13 * _smooth_square(math.sin(9.0 * t))) * math.sin(t),
        0.0,
        2 * math.pi - 0.001,
        420,
    ),
    "near-closed open nine-tooth gear profile",
)
add(
    "gear_rack_profile",
    param(
        lambda t: t,
        lambda t: 0.16 * _smooth_square(math.sin(2.0 * math.pi * 0.75 * t)),
        0.0,
        4.0,
        260,
    ),
    "linear rack teeth",
)

_t = Turtle(pos=(0.0, 1.6), heading=-math.pi / 2)
_t.straight(0.85, 3).arc(0.42, -3.6).arc(0.16, -1.1)
add_turtle("crane_hook", _t, "shank into hook sweep and tip flick")

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
_t.straight(1.6, 3).arc(0.30, math.pi).straight(1.35, 3).arc(0.21, math.pi)
_t.straight(1.05, 3).arc(0.12, math.pi).straight(0.7, 2)
add_turtle("paper_clip", _t, "nested U-turns joined by straights")

_n, _t0, _t1 = 320, 0.15, 8 * math.pi - 0.15
add(
    "coil_spring",
    param(lambda t: 0.22 * t, lambda t: 0.80 * math.sin(t), _t0, _t1, _n),
    "open sinusoidal coil-spring centerline",
)

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
_t.straight(1.0, 3).arc(0.45, math.pi / 2).straight(0.8, 3)
_t.arc(0.45, -math.pi / 2).straight(1.0, 3)
add_turtle("s_pipe_centerline", _t, "offset pipe route, two opposite bends")

_t = Turtle(pos=(0.0, 0.0), heading=0.35)
for r, a in [(0.5, 1.3), (0.3, -1.9), (0.6, 1.1), (0.25, -2.3), (0.45, 1.7)]:
    _t.arc(r, a).straight(0.3, 2)
add_turtle("exhaust_header", _t, "snaking exhaust runner")

add(
    "cam_with_notch",
    param(
        lambda t: (
            (1.0 + 0.30 * math.cos(t) - 0.26 * math.exp(-(((t - 4.1) / 0.18) ** 2)))
            * math.cos(t)
        ),
        lambda t: (
            (1.0 + 0.30 * math.cos(t) - 0.26 * math.exp(-(((t - 4.1) / 0.18) ** 2)))
            * math.sin(t)
        ),
        0.06,
        2 * math.pi - 0.06,
        300,
    ),
    "cam lobe with a concave dwell notch",
)
add(
    "corrugated_panel",
    param(
        lambda t: t, lambda t: 0.30 * math.sin(2.0 * math.pi * 0.8 * t), 0.0, 5.0, 180
    ),
    "sheet corrugation",
)

_t = Turtle(pos=(0.0, 0.0), heading=math.pi / 2)
for _ in range(4):
    _t.arc(0.16, -(math.pi - 0.5)).arc(0.16, math.pi - 0.5)
add_turtle("bellows_profile", _t, "deep accordion folds")

add(
    "wave_washer",
    param(
        lambda t: (1.0 + 0.09 * math.sin(6.0 * t)) * math.cos(t),
        lambda t: (1.0 + 0.09 * math.sin(6.0 * t)) * math.sin(t),
        0.04,
        2 * math.pi - 0.04,
        300,
    ),
    "six-wave washer rim",
)

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
_t.straight(1.5, 3).arc(0.10, math.pi / 2).straight(0.5, 2)
_t.arc(0.28, math.pi / 2).straight(0.5, 2).arc(0.10, math.pi / 2).straight(1.5, 3)
add_turtle("bracket_outline", _t, "L-bracket with inner and outer fillets")

add(
    "reflex_airfoil_camber",
    param(
        lambda t: t,
        lambda t: 0.11 * math.sin(math.pi * t) - 0.05 * math.sin(2 * math.pi * t),
        0.0,
        1.0,
        60,
    ),
    "reflexed camber line (aft inflection)",
)


def main() -> None:
    assert len(EXAMPLES) == 64, f"expected 64 examples, have {len(EXAMPLES)}"
    failures = []
    for i, (name, pts, note) in enumerate(EXAMPLES, start=1):
        try:
            comp = plot_nonconvex(i, name, pts, OUT, note=note)
            print(
                f"ok  {i:02d} {name:26s} {len(comp.points):4d} pts "
                f"{len(comp.runs):3d} runs {comp.n_segments:4d} segs"
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((i, name, exc))
            print(f"FAIL {i:02d} {name:26s} {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit(f"{len(failures)} example(s) failed")
    print(f"\nAll 64 nonconvex shape examples rendered to {OUT}")


if __name__ == "__main__":
    main()
