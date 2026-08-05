"""64 pathological direct-spline cases: extreme turns, features, scales.

Run:  python examples/cubic/generate_nonconvex_pathological.py
Output: examples/cubic/pathological/NN_name.png

Reliability gauntlet plus presentation pieces: near-reversal corners, deep
combs, smoothed fractals, oscillation extremes, loop tangles, coordinate
and curvature scale extremes, and classification-threshold riders.  Every
case fully constructs as one verified CubicPHSplineOpen, marks
auxiliary inflection points with translucent magenta crosses and 10
arc-equidistant points with red circles, and passes the independent
dense-polyline equidistance check.
"""

from __future__ import annotations

import math
import os

import numpy as np
from _composite import Turtle, param, plot_nonconvex

OUT = os.path.join(os.path.dirname(__file__), "pathological")

EXAMPLES: list[tuple[str, list, str, float]] = []


def add(name: str, pts, note: str = "", gap_tol=5e-3) -> None:
    EXAMPLES.append((name, pts, note, gap_tol))


def add_turtle(name: str, t: Turtle, note: str = "", gap_tol=5e-3) -> None:
    add(name, t.data(), note, gap_tol=gap_tol)


def zigzag(n_teeth, dx, height, x0=0.0, y0=0.0):
    pts = [[x0, y0]]
    for k in range(n_teeth):
        pts.append([x0 + (k + 0.5) * dx, y0 + height])
        pts.append([x0 + (k + 1.0) * dx, y0])
    return pts


def with_end_support(points):
    """Split both terminal spans to prescribe their intended line tangents."""
    out = [list(map(float, p)) for p in points]
    a, b = out[0], out[1]
    out.insert(1, [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])])
    a, b = out[-2], out[-1]
    out.insert(-1, [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])])
    return out


def rounded_polyline(points, radius_fraction=0.20, samples_per_corner=5):
    """Replace polygon corners by sampled quadratic fillets.

    Every retained straight section receives a midpoint.  Thus the example
    data explicitly identify the straight tangent instead of asking a free
    endpoint or one-span transition to infer it from a sharp corner.
    """
    a = np.asarray(points, dtype=np.float64)
    if a.shape[0] < 3:
        return a.tolist()
    before = []
    after = []
    for i in range(1, a.shape[0] - 1):
        vin = a[i] - a[i - 1]
        vout = a[i + 1] - a[i]
        lin = float(np.hypot(vin[0], vin[1]))
        lout = float(np.hypot(vout[0], vout[1]))
        trim = radius_fraction * min(lin, lout)
        before.append(a[i] - trim * vin / lin)
        after.append(a[i] + trim * vout / lout)

    out = [a[0].tolist(), (0.5 * (a[0] + before[0])).tolist(), before[0].tolist()]
    for i, (p0, p1) in enumerate(zip(before, after)):
        corner = a[i + 1]
        for k in range(1, samples_per_corner + 1):
            t = k / samples_per_corner
            q = (1.0 - t) ** 2 * p0 + 2.0 * (1.0 - t) * t * corner + t * t * p1
            out.append(q.tolist())
        if i + 1 < len(before):
            out.append((0.5 * (p1 + before[i + 1])).tolist())
            out.append(before[i + 1].tolist())
    out.extend([(0.5 * (after[-1] + a[-1])).tolist(), a[-1].tolist()])
    return out


# ---------------------------------------------------------------------------
# Turn extremes (1-16)
# ---------------------------------------------------------------------------

add(
    "sawtooth_near_pi",
    with_end_support(zigzag(20, 0.05, 1.0)),
    "apex turns pi - 0.05 rad; every span is handled by the main class's "
    "alternating convex sub-splines",
)

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for k in range(7):
    sgn = 1.0 if k % 2 == 0 else -1.0
    _t.straight(0.5, 2).arc(0.02, sgn * math.pi / 2).straight(0.5, 2)
    _t.arc(0.02, sgn * math.pi / 2)
add_turtle(
    "square_wave_tight_fillets",
    _t,
    "r = 0.02 fillets between 0.5 straights (chord ratio 1:60)",
)

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for _ in range(50):
    _t.straight(0.2, 1).corner(math.pi / 2).straight(0.2, 1).corner(-math.pi / 2)
add(
    "staircase_100_steps",
    rounded_polyline(_t.data(), radius_fraction=0.12, samples_per_corner=4),
    "100 supported treads and risers joined by small corner fillets",
)

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for _ in range(6):
    for a in (1, 1, -1, -1, -1, 1, 1):
        _t.straight(0.25, 1).corner(a * math.pi / 2)
    _t.straight(0.25, 1)
add(
    "greek_key_meander",
    rounded_polyline(_t.data(), radius_fraction=0.12, samples_per_corner=4),
    "classic meander with explicit straight runs and corner fillets",
)

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for _ in range(6):
    _t.straight(0.5, 2).arc(0.012, math.pi - 0.06).straight(0.5, 2)
    _t.arc(0.012, -(math.pi - 0.06))
add_turtle("comb_teeth_deep", _t, "deep teeth: r = 0.012 U-turns between 0.5 straights")

_ekg = [
    [0.0, 0.0],
    [0.8, 0.0],
    [1.6, 0.0],
    [1.72, 0.02],
    [1.80, 1.15],
    [1.88, -0.55],
    [1.96, 0.02],
    [2.1, 0.0],
    [2.9, 0.0],
    [3.7, 0.0],
    [3.82, 0.02],
    [3.90, 1.15],
    [3.98, -0.55],
    [4.06, 0.02],
    [4.2, 0.0],
    [5.0, 0.0],
    [5.8, 0.0],
]
add(
    "ekg_trace",
    rounded_polyline(_ekg, radius_fraction=0.08, samples_per_corner=5),
    "flat supported baseline with two tightly filleted QRS-like spikes",
)

rng = np.random.default_rng(11)
_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for _ in range(14):
    h = 0.3 + 1.4 * float(rng.random())
    w = 0.15 + 0.5 * float(rng.random())
    _t.corner(math.pi / 2).straight(h, 1).corner(-math.pi / 2).straight(w, 1)
    _t.corner(-math.pi / 2).straight(h, 1).corner(math.pi / 2).straight(w * 0.6, 1)
add(
    "city_skyline",
    rounded_polyline(_t.data(), radius_fraction=0.08, samples_per_corner=4),
    "seeded skyline with supported walls, roofs and small corner fillets",
)

_pts = [
    [0.0, 0.0],
    [0.55, -0.45],
    [0.18, -1.0],
    [0.82, -1.55],
    [0.42, -2.1],
    [1.0, -2.65],
]
add(
    "lightning_bolt",
    rounded_polyline(_pts, radius_fraction=0.10, samples_per_corner=5),
    "alternating bolt with supported strokes and tight corner fillets",
)

add(
    "porcupine_quills",
    with_end_support(zigzag(16, 0.35, 1.15)),
    "alternating +-2.84 rad turns, each alone in a 3-point run and "
    "interpolated as a full quill loop",
)

_sh = [[0.0, 0.0]]
for k in range(12):
    _sh.append([_sh[-1][0] + 0.18, 0.85])
    _sh.append([_sh[-1][0] + 0.72, 0.0])
add(
    "shark_teeth",
    rounded_polyline(_sh, radius_fraction=0.06, samples_per_corner=4),
    "asymmetric sawtooth with explicit straight faces and small root fillets",
)

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for k in range(5):
    _t.straight(0.25, 1).arc(0.09, -2.4).arc(0.16, 2.2 + 0.15 * k)
    _t.arc(0.16, 2.2 + 0.15 * k).arc(0.09, -2.4)
add_turtle("dripping_paint", _t, "pendulous drips: deep same-sign U runs")

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for k in range(4):
    sgn = 1.0 if k % 2 == 0 else -1.0
    for a in (2.02, 2.02):
        _t.arc(0.3, sgn * a)
add_turtle(
    "serpentine_max_pairs",
    _t,
    "each run holds an interior turn pair of 4.04 rad (bound 4.09691)",
)

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for k in range(6):
    sgn = 1.0 if k % 2 == 0 else -1.0
    _t.straight(1.0 - 0.13 * k, 2).arc(0.05, sgn * (math.pi - 0.02), n=9)
_t.straight(0.3, 2)
add_turtle(
    "hairpin_cascade",
    _t,
    "hairpins of pi - 0.02 rad with a supported final straight",
)

_star = []
for k in range(12):
    a = 2.0 * math.pi * k / 12.0
    _star.append([math.cos(a), math.sin(a)])
    a2 = a + math.pi / 12.0
    _star.append([1.9 * math.cos(a2), 1.9 * math.sin(a2)])
_star.append(
    [
        0.02 * _star[-1][0] + 0.98 * _star[0][0],
        0.02 * _star[-1][1] + 0.98 * _star[0][1],
    ]
)
add(
    "star_zigzag_radial",
    with_end_support(_star),
    "near-closed open radial star with supported seam tangents",
)

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for _ in range(3):
    _t.straight(1.0, 4)
    for _ in range(6):
        _t.corner(math.pi / 2).straight(0.02, 1).corner(-math.pi / 2)
        _t.straight(0.02, 1).corner(-math.pi / 2).straight(0.02, 1)
        _t.corner(math.pi / 2).straight(0.02, 1)
add_turtle("battlement_fine", _t, "0.02-scale crenellations on 1.0-scale walls")

_t = Turtle(pos=(0.0, 0.0), heading=0.35)
for k in range(18):
    sgn = 1.0 if k % 2 == 0 else -1.0
    _t.arc(1.6, 0.28).corner(sgn * 2.1)
add_turtle("pinking_shears_edge", _t, "triangle wave riding a slow arc")

# ---------------------------------------------------------------------------
# Smoothed fractals (17-24)
# ---------------------------------------------------------------------------


def l_system(axiom: str, rules: dict[str, str], depth: int) -> str:
    s = axiom
    for _ in range(depth):
        s = "".join(rules.get(ch, ch) for ch in s)
    return s


def trace(cmds: str, angle: float, step: float = 1.0) -> list:
    h = 0.0
    pts = [[0.0, 0.0]]
    for ch in cmds:
        if ch == "F":
            p = pts[-1]
            pts.append([p[0] + step * math.cos(h), p[1] + step * math.sin(h)])
        elif ch == "+":
            h += angle
        elif ch == "-":
            h -= angle
    return pts


add(
    "koch_curve_smoothed",
    trace(l_system("F", {"F": "F+F--F+F"}, 3), math.pi / 3.0),
    "level-3 Koch curve interpolated smoothly (64 chords)",
)
add(
    "dragon_curve_smoothed",
    rounded_polyline(
        trace(
            l_system("FX", {"X": "X+YF+", "Y": "-FX-Y"}, 6)
            .replace("X", "")
            .replace("Y", ""),
            math.pi / 2.0,
        ),
        radius_fraction=0.16,
        samples_per_corner=5,
    ),
    "level-6 dragon curve with supported links and rounded corners",
)
_hilbert_vertices = trace(
    l_system("A", {"A": "-BF+AFA+FB-", "B": "+AF-BFB-FA+"}, 3)
    .replace("A", "")
    .replace("B", ""),
    math.pi / 2.0,
)
add(
    "hilbert_curve_smoothed",
    rounded_polyline(_hilbert_vertices, radius_fraction=0.18, samples_per_corner=5),
    "level-3 Hilbert curve with explicit straight runs and rounded corners",
)
_levy = param(
    lambda t: t + 0.08 * math.sin(7.0 * t),
    lambda t: 0.60 * math.sin(3.0 * t),
    0.0,
    4.0,
    260,
)
add(
    "levy_c_smoothed",
    _levy,
    "open Levy-C-inspired multi-scale wave",
)
add(
    "cesaro_sweep",
    rounded_polyline(
        trace(l_system("F", {"F": "F+F--F+F"}, 3), 1.45),
        radius_fraction=0.12,
        samples_per_corner=5,
    ),
    "Cesaro sweep with supported links and tight corner fillets",
)
add(
    "minkowski_sausage",
    rounded_polyline(
        trace(l_system("F", {"F": "F+F-F-FF+F+F-F"}, 2), math.pi / 2.0),
        radius_fraction=0.14,
        samples_per_corner=5,
    ),
    "level-2 Minkowski sausage with supported links and rounded corners",
)
_cantor = [[0.0, 0.0]]
for a, b in [
    (0.0, 1.0),
    (2.0, 3.0),
    (6.0, 7.0),
    (8.0, 9.0),
    (18.0, 19.0),
    (20.0, 21.0),
    (24.0, 25.0),
    (26.0, 27.0),
]:
    _cantor += [[a, 0.0], [a + 0.15, 0.9], [a + 0.5, 1.15], [a + 0.85, 0.9], [b, 0.0]]
_cantor = [p for i, p in enumerate(_cantor) if i == 0 or p != _cantor[i - 1]]
add("cantor_comb", _cantor, "bumps at Cantor-set gaps: spacing spans 1 to 9 units")
add(
    "terdragon_smoothed",
    param(
        lambda t: t,
        lambda t: 0.4 * math.sin(5.0 * t) + 0.1 * math.sin(11.0 * t),
        0.0,
        4.0,
        260,
    ),
    "terdragon-inspired nested-frequency sweep",
)

# ---------------------------------------------------------------------------
# Oscillation extremes (25-34)
# ---------------------------------------------------------------------------

add(
    "chirp_rising",
    param(
        lambda t: t,
        lambda t: 0.42 * math.sin(2.0 * math.pi * (0.4 * t + 0.55 * t * t)),
        0.0,
        4.0,
        1400,
    ),
    "frequency rises 12x: late turns approach the reversal regime",
)
add(
    "amplitude_blowup",
    param(
        lambda t: t,
        lambda t: 0.004 * math.exp(1.25 * t) * math.sin(7.0 * t),
        0.0,
        4.5,
        500,
    ),
    "envelope grows 280x across the span",
)
add(
    "damped_to_1e-8",
    param(
        lambda t: t,
        lambda t: 0.5 * math.exp(-1.8 * t) * math.sin(6.0 * t),
        0.0,
        10.0,
        620,
    ),
    "amplitude decays to 1e-8: late runs are near the collinear threshold",
)
add(
    "flat_s_amplitude_1e-9",
    param(lambda t: t, lambda t: 1e-9 * math.sin(2.0 * math.pi * t), 0.0, 1.0, 40),
    "one sine period at amplitude 1e-9 on unit length",
)
rng = np.random.default_rng(20260804)
_noise = 1e-4 * rng.standard_normal(1200)
add(
    "noisy_sine_1200pts",
    [[i / 120.0, math.sin(i / 120.0 * 2.2) + float(_noise[i])] for i in range(1200)],
    "1200 samples with 1e-4 noise: dense inflection churn",
)
_seis = []
for x in np.linspace(0.0, 10.0, 900):
    y = 0.0
    for center, amp, phase in (
        (1.25, 0.75, 0.2),
        (3.35, 0.95, 1.1),
        (5.75, 1.10, -0.7),
        (8.15, 0.85, 0.6),
    ):
        q = x - center
        envelope = math.exp(-((q / 0.36) ** 2))
        y += (
            amp
            * envelope
            * (
                0.64 * math.sin(34.0 * q + phase)
                + 0.25 * math.sin(53.0 * q - 0.4 * phase)
            )
        )
    _seis.append([float(x), y])
add(
    "seismograph",
    _seis,
    "four smooth high-frequency bursts separated by quiet intervals",
    gap_tol=8e-3,
)
add(
    "beat_pattern",
    param(
        lambda t: t,
        lambda t: 0.28 * (math.sin(11.0 * t) + math.sin(13.0 * t)),
        0.0,
        6.3,
        620,
    ),
    "two close frequencies beating",
)
add(
    "flattened_sine",
    param(
        lambda t: t,
        lambda t: (
            0.5
            * math.sin(2.5 * t)
            / math.sqrt(0.04 + math.sin(2.5 * t) ** 2)
            * math.sqrt(1.04)
        ),
        0.0,
        6.0,
        400,
    ),
    "square-ish wave: flat crests, steep flanks",
)
add(
    "chirp_at_offset_1e9",
    param(
        lambda t: 1e9 + t,
        lambda t: -1e9 + 0.4 * math.sin(2.0 * math.pi * (0.4 * t + 0.4 * t * t)),
        0.0,
        4.0,
        900,
    ),
    "rising chirp displaced 1e9 units from the origin",
)
_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for k in range(11):
    sgn = 1.0 if k % 2 == 0 else -1.0
    _t.arc(0.25 + 0.55 * abs(math.sin(1.1 * k)), sgn * (1.6 + 0.9 * math.sin(0.7 * k)))
add_turtle("modulated_meander", _t, "radius- and angle-modulated meander")

# ---------------------------------------------------------------------------
# Loops and tangles (35-44)
# ---------------------------------------------------------------------------

_n, _t0, _t1 = 360, 0.15, 10 * math.pi - 0.15
add(
    "loop_the_loop",
    param(lambda t: 0.30 * t, lambda t: 1.20 * math.sin(t), _t0, _t1, _n),
    "five high-amplitude open oscillations",
)
rng = np.random.default_rng(77)
_a = [float(rng.uniform(0.4, 1.0)) for _ in range(3)]
_ph = [float(rng.uniform(0, 2 * math.pi)) for _ in range(4)]
_n = 420
add(
    "overhand_scribble",
    param(
        lambda t: (
            _a[0] * math.sin(t + _ph[0]) + 0.55 * _a[1] * math.sin(2 * t + _ph[1])
        ),
        lambda t: _a[1] * math.cos(t + _ph[2]) + 0.5 * _a[2] * math.sin(3 * t + _ph[3]),
        0.05,
        2 * math.pi - 0.05,
        _n,
    ),
    "seeded Fourier scribble with self-crossings",
)
_out = [
    [(0.25 + 0.11 * t) * math.cos(t), (0.25 + 0.11 * t) * math.sin(t)]
    for t in np.linspace(0.0, 3.4 * math.pi, 120)
]
_back = [
    [(0.25 + 0.11 * t + 0.28) * math.cos(t), (0.25 + 0.11 * t + 0.28) * math.sin(t)]
    for t in np.linspace(3.4 * math.pi, 0.0, 120)
]
add(
    "there_and_back_spiral",
    _out + _back[1:],
    "archimedean spiral out, reversal, nested return lane",
)

rng = np.random.default_rng(1234)
_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for _ in range(24):
    _t.arc(
        0.22 + 0.5 * float(rng.random()),
        float(rng.uniform(0.8, 2.6)) * (1 if rng.random() < 0.55 else -1),
    )
add_turtle("tangle_walk", _t, "seeded arc walk with many crossings")

_n = 500
add(
    "spirograph_dense",
    param(
        lambda t: (1.0 + 0.28 * math.cos(13.0 * t)) * math.cos(t),
        lambda t: (1.0 + 0.28 * math.cos(13.0 * t)) * math.sin(t),
        0.02,
        2 * math.pi - 0.02,
        _n,
    ),
    "dense open thirteen-lobe spirograph rim",
)
_n = 420
add(
    "rose_5_petals",
    param(
        lambda t: (1.0 + 0.4 * math.cos(5.0 * t)) * math.cos(t),
        lambda t: (1.0 + 0.4 * math.cos(5.0 * t)) * math.sin(t),
        0.04,
        2 * math.pi - 0.04,
        _n,
    ),
    "five-petal open rose rim without a repeated center point",
)
_t = Turtle(pos=(0.0, 0.0), heading=0.55)
for k in range(6):
    sgn = 1.0 if k % 2 == 0 else -1.0
    _t.arc(0.30, sgn * 4.9, n=18).straight(0.22, 1)
add_turtle("loop_garland", _t, "alternating 280-degree loops (cursive ee)")

_t = Turtle(pos=(0.0, 0.0), heading=1.1)
_t.arc(0.55, 4.3, n=16).arc(0.30, -2.2).arc(0.55, 4.3, n=16).arc(0.30, -2.2)
_t.arc(0.55, 4.3, n=16)
add_turtle("pretzel_knot", _t, "three big loops linked by reverse curls")

_n = 480
add(
    "torus_knot_3_4",
    param(
        lambda t: t + 0.15 * math.sin(4.0 * t),
        lambda t: 0.7 * math.sin(3.0 * t),
        0.0,
        3 * math.pi,
        _n,
    ),
    "open 3:4 braided-wave projection",
)
_n, _t0, _t1 = 420, 0.12, 12 * math.pi - 0.12
add(
    "collapsed_coil",
    param(
        lambda t: 0.11 * t,
        lambda t: 0.85 * math.sin(t) * (1.0 + 0.006 * t),
        _t0,
        _t1,
        _n,
    ),
    "slow-advancing open coil with increasing amplitude",
)

# ---------------------------------------------------------------------------
# Scale extremes (45-54)
# ---------------------------------------------------------------------------

_macro = [
    [math.cos(a), math.sin(a)] for a in np.linspace(math.pi, math.pi / 2 + 0.03, 40)
]
_apex = _macro[-1]
_micro = []
for k in range(6):
    _micro.append(
        [_apex[0] + (k + 0.6) * 3e-5, _apex[1] + 2.2e-5 * (1 if k % 2 == 0 else -1)]
    )
_macro2 = [[math.cos(a), math.sin(a)] for a in np.linspace(math.pi / 2 - 0.03, 0.0, 40)]
add(
    "micro_detail_on_macro",
    _macro + _micro + _macro2,
    "3e-5-scale zigzag inserted at the apex of a unit arc",
)

add(
    "offset_1e9_wiggle",
    param(
        lambda t: 1e9 + t,
        lambda t: 1e9 + 0.3 * math.sin(3.0 * t) * math.sin(0.8 * t),
        0.0,
        8.0,
        300,
    ),
    "modulated wiggle a billion units from the origin",
)

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for k in range(10):
    sgn = 1.0 if (k // 2) % 2 == 0 else -1.0
    _t.arc(0.6, sgn * 0.30, n=3).arc(2e-3, sgn * 0.30, n=3)
add_turtle(
    "chord_jump_chain_300x",
    _t,
    "arcs alternating 300x in radius inside same-sign runs -- the "
    "largest jump whose residual verifies below the strict 1e-11 "
    "gate (1000x hits the conditioning floor and is refused)",
)

_scales: list[list[float]] = []
_base = [0.0, 0.0]
for s, turns in [(1.0, 3.0 * math.pi), (1e-3, 3.0 * math.pi), (1e-6, 2.5 * math.pi)]:
    if _scales:
        # Place the next spiral ahead of and beside the previous end so
        # its winding stays clear of the incoming chords.
        prev, prev2 = _scales[-1], _scales[-2]
        dx, dy = prev[0] - prev2[0], prev[1] - prev2[1]
        nrm = math.hypot(dx, dy)
        _base = [
            prev[0] + (dx / nrm) * 1.2 * s - (dy / nrm) * 1.6 * s,
            prev[1] + (dy / nrm) * 1.2 * s + (dx / nrm) * 1.6 * s,
        ]
    for t in np.linspace(0.15, turns, 70):
        r = s * 0.4 * math.exp(-0.25 * t)
        _scales.append([_base[0] + r * math.cos(-t), _base[1] + r * math.sin(-t)])
add(
    "three_scale_spiral_chain",
    _scales,
    "inward spirals nested at scales 1, 1e-3, 1e-6",
    gap_tol=8e-3,
)

add(
    "zigzag_at_1e100",
    with_end_support([[k * 1e100, (1e100 if k % 2 else 0.0)] for k in range(14)]),
    "sawtooth with 1e100-unit teeth and supported terminal faces",
)
add(
    "meander_at_1e-100",
    [[k * 2e-100, 1e-100 * math.sin(1.1 * k)] for k in range(30)],
    "gentle meander at coordinate scale 1e-100",
)
add(
    "long_thin_ribbon",
    param(
        lambda t: 1000.0 * t, lambda t: math.sin(2.0 * math.pi * 2.0 * t), 0.0, 1.0, 260
    ),
    "aspect ratio 500:1 (display normalizes the view)",
)
add(
    "staircase_at_1e150",
    rounded_polyline(
        [[(k // 2) * 1e150 + (k % 2) * 1e150, (k // 2) * 1e150] for k in range(1, 21)],
        radius_fraction=0.10,
        samples_per_corner=4,
    ),
    "staircase with 1e150-unit supported treads and corner fillets",
)
rng = np.random.default_rng(99)
_mega = [
    [
        i * 0.004,
        0.6 * math.sin(i * 0.02)
        + 0.25 * math.sin(i * 0.11 + 1.0)
        + 3e-3 * float(rng.standard_normal()),
    ]
    for i in range(2500)
]
add(
    "megapolyline_2500",
    _mega,
    "2500 points, three interleaved wave scales",
    gap_tol=8e-3,
)

_needles = [[0.0, 0.0]]
for k in range(8):
    x = 0.5 + 0.6 * k
    _needles += [[x, 0.0], [x + 1e-4, 1.0], [x + 2e-4, 0.0]]
_needles.append([5.4, 0.0])
add(
    "needle_spikes",
    rounded_polyline(_needles, radius_fraction=0.04, samples_per_corner=5),
    "unit-height needles with supported baseline and tightly filleted tips",
    gap_tol=8e-3,
)

# ---------------------------------------------------------------------------
# Threshold riders (55-64)
# ---------------------------------------------------------------------------

_alln = 30
_all_inf = [[0.0, 0.0]]
for k in range(1, _alln):
    _all_inf.append([k * 1.0, ((-1) ** k) * 5e-13 * k])
add(
    "all_inflections_1e-12",
    _all_inf,
    "turns ~1e-12 rad alternating sign at every point: every interior "
    "point splits, 29 two-point runs",
)

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for k in range(5):
    sgn = 1.0 if k % 2 == 0 else -1.0
    _t.arc(0.28, sgn * 2.045).arc(0.28, sgn * 2.045)
add_turtle(
    "pair_sum_ladder",
    _t,
    "interior pairs at 4.09 of 4.09691 rad, alternating sign blocks",
)

# Alternating near-reversal folds, partitioned only by section-22 joints.
_pts = [
    [-0.4, -0.1],
    [0.0, 0.0],
    [1.0, 0.0],
    [0.15, 0.12],
    [1.2, 0.27],
    [0.25, 0.45],
    [1.3, 0.66],
    [0.35, 0.9],
]
add(
    "almost_reversal_folds",
    rounded_polyline(_pts, radius_fraction=0.10, samples_per_corner=7),
    "alternating near-reversal folds with sampled fillets and supported ends",
)

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for k in range(5):
    sgn = 1.0 if k % 2 == 0 else -1.0
    _t.straight(0.6, 3).arc(0.4, sgn * 1.9)
add_turtle(
    "zero_turn_boundaries",
    _t,
    "exactly collinear spans between arcs (straight/curved splits)",
)

_thr = [[0.0, 0.0]]
for k in range(1, 24):
    mag = 5e-13 if k % 2 else 1e-11
    _thr.append([k * 1.0, _thr[-1][1] + mag * k * (1 if (k // 2) % 2 else -1)])
add(
    "classification_threshold_mix",
    _thr,
    "turns straddling the numerically-zero threshold 2.27e-13",
)

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for k in range(5):
    sgn = 1.0 if k % 2 == 0 else -1.0
    _t.arc(0.35, sgn * 2.9, n=4).straight(0.25, 1)
add_turtle(
    "boundary_clamp_gauntlet",
    _t,
    "every crescent run engages the boundary-tangent clamp",
)

_t = Turtle(pos=(0.0, 0.0), heading=0.0)
for k in range(6):
    sgn = 1.0 if k % 2 == 0 else -1.0
    _t.arc(0.5, sgn * 2.7, n=3).arc(0.5, sgn * 0.1, n=2)
add_turtle(
    "solver_asymmetry_chain",
    _t,
    "2.7 / 0.1 rad turn alternation inside each sign block",
)

add(
    "micro_garland_kappa_1e6",
    param(lambda t: 3.0e-6 * t, lambda t: 1.2e-6 * math.sin(t), 0.0, 7 * math.pi, 240),
    "micron-scale open garland: curvature reaches order 1e6",
)

_sleep = [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]
_sleep += [[1.2 + 0.05 * k, 1.5e-7 * math.sin(0.9 * k)] for k in range(1, 14)]
_sleep += [[2.1, 0.0], [2.6, 0.0], [3.1, 0.0]]
_sleep += [
    [3.2 + 0.03 * k, -2e-7 * (1 if k % 2 else 2) * (k % 3)] for k in range(1, 10)
]
_sleep += [[3.8, 0.0], [4.4, 0.0]]
add(
    "sleeping_giant",
    _sleep,
    "1e-7-amplitude features embedded in an exactly flat line",
    gap_tol=8e-3,
)

_g = Turtle(pos=(0.0, 0.0), heading=0.0)
for _ in range(8):
    _g.straight(0.3, 1).corner(math.pi / 2).straight(0.3, 1).corner(-math.pi / 2)
for k in range(4):
    sgn = 1.0 if k % 2 == 0 else -1.0
    _g.arc(0.3, sgn * 4.7, n=17).straight(0.25, 1)
for k in range(30):
    sgn = 1.0 if k % 2 == 0 else -1.0
    _g.arc(0.10 + 0.004 * k, sgn * (1.8 + 0.03 * k), n=7)
_g.straight(1.0, 4)
for k in range(6):
    sgn = 1.0 if k % 2 == 0 else -1.0
    _g.arc(0.02, sgn * (math.pi - 0.1), n=8).straight(0.35, 2)
add_turtle(
    "the_gauntlet", _g, "stairs, loops, accelerating serpentine, flat run, hairpins"
)


def main() -> None:
    assert len(EXAMPLES) == 64, f"expected 64 cases, have {len(EXAMPLES)}"
    failures = []
    for i, (name, pts, note, tol) in enumerate(EXAMPLES, start=1):
        try:
            comp = plot_nonconvex(i, name, pts, OUT, note=note, gap_tol=tol)
            print(
                f"ok  {i:02d} {name:30s} {len(comp.points):4d} pts "
                f"{len(comp.runs):3d} runs {comp.n_segments:4d} segs"
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((i, name, exc))
            print(f"FAIL {i:02d} {name:30s} {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit(f"{len(failures)} case(s) failed")
    print(f"\nAll 64 pathological nonconvex cases rendered to {OUT}")


if __name__ == "__main__":
    main()
