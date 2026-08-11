"""Render 32 open PH B-spline exact-offset studies into this folder.

Run from the repository root:
    python examples/bspline_offset/generate_examples.py

Use ``--check-only`` to construct and verify every offset without drawing.
Each case builds a :class:`PHBSplineOpen` (several at higher continuity
orders), requests exact rational NURBS offsets of degree
``4 * preimage_degree + 1``, and verifies them against
``z(u) + d * N_L(u)`` before publishing an image.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from offset_common import (  # noqa: E402
    OffsetCase,
    catmull_rom,
    meander,
    open_curve,
    run,
    seeded_walk,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ph_spline import PHBSplineOpen  # noqa: E402

OUT = Path(__file__).resolve().parent
CASES: list[OffsetCase] = []


def add(name, points, distances, note, category, **kwargs):
    CASES.append(OffsetCase(name, points, tuple(distances), note, category,
                            **kwargs))


# -- manufacturing ----------------------------------------------------------

add(
    "five_axis_flank_passes",
    open_curve(lambda t: (t, 0.55 * np.sin(0.8 * t) + 0.08 * t), 0.0, 6.4, 12),
    (0.3, 0.6, 0.9, 1.2),
    "flank-milling stepover passes from a G4 drive curve",
    "manufacturing",
    kwargs={"g_order": 4},
)
add(
    "composite_tape_courses",
    open_curve(
        lambda t: (t, 0.85 * np.sin(0.62 * t) * (1.0 - 0.07 * t)), 0.0, 7.2, 13
    ),
    (0.32, 0.64, 0.96, 1.28),
    "automated tape-laying courses beside a composite master course",
    "manufacturing",
)
add(
    "fiber_steering_bands",
    catmull_rom(
        [(0.0, 0.0), (1.3, 0.75), (2.6, 0.55), (3.8, 1.25), (5.1, 1.0),
         (6.3, 1.7)],
        4, closed=False,
    ),
    (-0.3, 0.3, 0.6),
    "steering-radius audit bands about a fiber placement course",
    "manufacturing",
)
add(
    "cnc_engraving_hatches",
    meander(6.0, lambda x: 0.5 * np.sin(1.35 * x) + 0.14 * np.sin(3.4 * x), 21),
    (0.14, 0.28, 0.42, 0.56, 0.7),
    "hatch-fill passes stepped from an engraved border stroke",
    "manufacturing",
)
add(
    "robot_polishing_passes",
    open_curve(
        lambda t: (2.9 * np.cos(t), 1.75 * np.sin(t)), -0.4, 1.95, 13
    ),
    (0.22, 0.44, 0.66),
    "compliant polishing passes inside an elliptic panel edge",
    "manufacturing",
    kwargs={"g_order": 3},
)
add(
    "waterline_machining_shells",
    catmull_rom(
        [(0.0, 1.9), (0.5, 0.9), (1.4, 0.25), (2.8, 0.0), (4.2, 0.15),
         (5.3, 0.7)],
        4, closed=False,
    ),
    (0.24, 0.5, 0.78),
    "waterline machining shells offset from a die sweep curve",
    "manufacturing",
)

# -- CFD and physics --------------------------------------------------------

add(
    "nacelle_inlet_layers",
    open_curve(
        lambda t: (t, 0.6 * np.sqrt(np.maximum(t, 0.0)) - 0.05 * t * t),
        0.0, 4.6, 12,
    ),
    (0.06, 0.13, 0.22, 0.34, 0.5),
    "growing viscous inflation layers over a nacelle inlet lip line",
    "cfd",
    kwargs={"g_order": 4},
)
add(
    "wake_survey_stations",
    meander(8.0, lambda x: 0.9 * np.exp(-0.35 * (x - 4.0) ** 2), 17),
    (-0.35, -0.7, -1.05),
    "wake-rake survey stations below a displacement-thickness hump",
    "cfd",
)
add(
    "beamline_envelope_bands",
    open_curve(
        lambda t: (t, 0.34 * np.sin(1.05 * t) * np.exp(-0.06 * t)), 0.0, 8.0, 15
    ),
    (0.2, -0.2, 0.42, -0.42),
    "vacuum-envelope tolerance bands about a particle beamline axis",
    "physics",
    kwargs={"g_order": 4},
)
add(
    "undulator_field_shells",
    meander(6.5, 0.4, 27, frequency=4.0),
    (0.17, -0.17),
    "pole-face shells about an undulator sinusoidal axis",
    "physics",
)
add(
    "damped_oscillation_envelopes",
    open_curve(
        lambda t: (t, 1.25 * np.exp(-0.35 * t) * np.cos(2.4 * t)), 0.0, 6.0, 25
    ),
    (0.16, -0.16),
    "constant-clearance envelopes hugging a damped oscillation trace",
    "physics",
)
add(
    "cryo_transfer_insulation",
    catmull_rom(
        [(0.0, 0.0), (1.2, 0.4), (2.2, 1.4), (3.6, 1.6), (4.9, 1.0),
         (6.1, 1.25)],
        4, closed=False,
    ),
    (0.18, 0.38, 0.6),
    "vacuum-jacket insulation layers along a cryogen transfer line",
    "physics",
)

# -- space engineering ------------------------------------------------------

add(
    "docking_approach_cone",
    open_curve(lambda t: (t, 0.055 * t * t), -3.2, 3.2, 15),
    (0.25, 0.55, 0.9),
    "approach-corridor boundaries above a docking axis parabola",
    "space",
    kwargs={"g_order": 4},
)
add(
    "debris_avoidance_bands",
    open_curve(
        lambda t: ((4.1 - 0.35 * t) * np.cos(t), (3.0 - 0.26 * t) * np.sin(t)),
        0.2, 2.9, 17,
    ),
    (0.3, 0.62, -0.3),
    "conjunction screening bands about a decaying orbit arc",
    "space",
)
add(
    "rover_slope_contours",
    seeded_walk(4257, 22, step=0.8, turn_scale=0.34),
    (0.45, 0.95, 1.5),
    "traversability contour bands downslope of a rover route",
    "space",
)
add(
    "aerobraking_corridor",
    open_curve(lambda t: (2.9 * np.sinh(t), 1.9 * np.cosh(t) - 1.9), -1.0, 1.0, 13),
    (0.28, -0.28),
    "entry-corridor bounds about an aerobraking hyperbolic arc",
    "space",
    kwargs={"g_order": 3},
)

# -- mobility and logistics -------------------------------------------------

add(
    "high_speed_rail_clearance",
    catmull_rom(
        [(0.0, 0.0), (1.8, 0.25), (3.4, 1.05), (5.2, 1.25), (7.0, 0.7),
         (8.6, 0.9)],
        4, closed=False,
    ),
    (-0.5, -0.25, 0.25, 0.5),
    "structure gauge and walkway clearances about a jerk-limited alignment",
    "mobility",
    kwargs={"g_order": 4},
)
add(
    "maglev_guideway_bands",
    open_curve(
        lambda t: (t, 1.05 * np.sin(0.55 * t) + 0.1 * np.sin(1.7 * t)),
        0.0, 9.0, 17,
    ),
    (-0.32, 0.32),
    "levitation-gap guideway bands beside a maglev centerline",
    "mobility",
    kwargs={"g_order": 4},
)
add(
    "roller_coaster_rail_pair",
    open_curve(
        lambda t: (t, 1.5 * np.exp(-0.5 * (t - 2.1) ** 2)
                   + 0.75 * np.exp(-0.8 * (t - 4.6) ** 2)),
        0.0, 6.6, 21,
    ),
    (-0.22, 0.22),
    "twin rail traces about a coaster spine with two crests",
    "mobility",
    kwargs={"g_order": 4},
)
add(
    "port_approach_fairway",
    seeded_walk(808, 20, step=0.95, turn_scale=0.28),
    (0.6, 1.25, -0.6, -1.25),
    "dredged fairway and anchorage margins about an approach route",
    "logistics",
)
add(
    "baggage_conveyor_guides",
    catmull_rom(
        [(0.0, 0.0), (1.1, 0.65), (2.4, 0.5), (3.3, 1.3), (4.6, 1.15),
         (5.5, 0.45), (6.7, 0.6)],
        4, closed=False,
    ),
    (-0.2, 0.2),
    "side-guide rails about a baggage conveyor centerline",
    "logistics",
)
add(
    "drone_altitude_corridor",
    seeded_walk(1102, 24, step=0.85, turn_scale=0.3),
    (0.4, 0.85, -0.4, -0.85),
    "segregated drone corridor bands about a survey flight path",
    "logistics",
)
add(
    "subsea_trench_walls",
    meander(9.5, lambda x: 1.2 * np.sin(0.62 * x + 0.4) + 0.25 * np.sin(1.7 * x), 23),
    (-0.4, 0.4, 0.85),
    "trench walls and spoil berm along a subsea cable route",
    "logistics",
)
add(
    "wind_farm_cable_clearance",
    seeded_walk(77, 21, step=0.9, turn_scale=0.35),
    (0.5, 1.05),
    "burial clearance strips beside an inter-array cable route",
    "logistics",
)

# -- design and biomedical --------------------------------------------------

add(
    "ski_jump_inrun_shells",
    open_curve(
        lambda t: (t, 2.3 * np.exp(-0.55 * t) - 0.16 * t), 0.0, 6.0, 15
    ),
    (-0.16, -0.34, -0.55),
    "ice and structure shells beneath a ski-jump inrun profile",
    "design",
)
add(
    "stent_strut_envelope",
    meander(5.4, 0.42, 29, frequency=4.5),
    (0.1, -0.1),
    "strut-width envelope about a stent cell midline",
    "biomedical",
)
add(
    "cochlear_electrode_clearance",
    open_curve(
        lambda t: ((0.42 + 0.3 * t) * np.cos(t), (0.42 + 0.3 * t) * np.sin(t)),
        0.0, 5.2, 23,
    ),
    (0.1, -0.1),
    "insertion clearance about a cochlear electrode spiral",
    "biomedical",
)
add(
    "prosthetic_socket_lines",
    catmull_rom(
        [(0.0, 2.2), (0.42, 1.2), (0.5, 0.2), (0.95, -0.75), (1.9, -1.25),
         (3.0, -1.35)],
        4, closed=False,
    ),
    (0.14, 0.3, 0.5),
    "liner and socket wall lines built from a residual-limb profile",
    "biomedical",
)

# -- abstract ---------------------------------------------------------------

add(
    "lissajous_ribbon",
    open_curve(
        lambda t: (2.4 * np.sin(1.0 * t + 0.35), 1.5 * np.sin(2.0 * t)),
        0.12, 2.95, 25,
    ),
    (0.14, -0.14),
    "constant-width ribbon about one lobe of a Lissajous figure",
    "abstract",
    kwargs={"g_order": 3},
)
add(
    "confidence_band_walk",
    seeded_walk(31415, 26, step=0.8, turn_scale=0.42),
    (0.35, 0.75, -0.35, -0.75),
    "one- and two-sigma bands about an empirical random-walk fit",
    "abstract",
)
def _clothoid_points(t1: float, count: int) -> list[list[float]]:
    result = []
    for x in np.linspace(0.0, t1, count):
        s = np.linspace(0.0, x, 160)
        result.append([
            float(np.trapezoid(np.cos(0.85 * s * s), s)),
            float(np.trapezoid(np.sin(0.85 * s * s), s)),
        ])
    return result


add(
    "euler_spiral_offsets",
    _clothoid_points(3.1, 21),
    (0.16, 0.34, -0.16),
    "clothoid transition curve with rational offsets on both sides",
    "abstract",
    kwargs={"g_order": 3},
)
add(
    "cusp_gallery_sweep",
    open_curve(lambda t: (2.3 * np.cos(t), 1.5 * np.sin(t)), 0.3, 2.85, 17),
    (0.8, 1.5, 2.3),
    "inward sweep past the curvature radius: cusps and loops kept exactly",
    "abstract",
)


def build(case: OffsetCase) -> PHBSplineOpen:
    return PHBSplineOpen(case.points, **case.kwargs)


if __name__ == "__main__":
    run(CASES, build, OUT, "open PH B-spline source")
