"""Render 32 open cubic-PH exact-offset studies into this folder.

Run from the repository root:
    python examples/cubic_offset/generate_examples.py

Use ``--check-only`` to construct and verify every offset without drawing.
Each case builds a :class:`CubicPHSplineOpen`, requests several exact
rational NURBS offsets, and verifies them against ``r(u) + d * N_L(u)``
before publishing an image.
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

from ph_spline import CubicPHSplineOpen  # noqa: E402

OUT = Path(__file__).resolve().parent
CASES: list[OffsetCase] = []


def add(name, points, distances, note, category, **kwargs):
    CASES.append(OffsetCase(name, points, tuple(distances), note, category,
                            **kwargs))


# -- manufacturing ----------------------------------------------------------

add(
    "cnc_face_milling_pass",
    open_curve(lambda t: (t, 0.35 * np.sin(0.9 * t) + 0.05 * t), 0.0, 6.0, 14),
    (0.35, 0.7, 1.05, 1.4),
    "one-sided stepover fan for a face-milling boundary pass",
    "manufacturing",
)
add(
    "tool_radius_compensation",
    open_curve(lambda t: (t, 0.9 * np.sin(t) / (1.0 + 0.16 * t * t)), 0.0, 5.6, 15),
    (-0.24, 0.24),
    "left and right cutter-radius compensation of a programmed contour",
    "manufacturing",
)
add(
    "laser_kerf_compensation",
    catmull_rom(
        [(0.0, 0.0), (1.1, 0.7), (2.3, 0.15), (3.2, 0.9), (4.4, 0.35),
         (5.3, 1.05)],
        6, closed=False,
    ),
    (-0.09, 0.09),
    "small symmetric kerf allowance about a laser cutting path",
    "manufacturing",
)
add(
    "waterjet_taper_shells",
    open_curve(lambda t: (2.6 * np.cos(t), 1.9 * np.sin(t)), -0.35, 1.85, 12),
    (0.12, 0.26, 0.42, 0.6),
    "graded taper-compensation shells along an elliptic waterjet cut",
    "manufacturing",
)
add(
    "weld_multipass_beads",
    open_curve(lambda t: (t, 0.55 * np.cosh(0.6 * (t - 2.6)) - 0.55), 0.0, 5.2, 13),
    (0.16, 0.32, 0.48),
    "stacked weld-bead passes above a groove root line",
    "manufacturing",
)
add(
    "wire_edm_roughing_skims",
    catmull_rom(
        [(0.0, 0.2), (1.0, 0.85), (2.1, 0.3), (3.0, 1.1), (4.2, 0.75),
         (5.0, 1.35)],
        5, closed=False,
    ),
    (-0.3, -0.18, -0.09, -0.035),
    "diminishing skim offsets converging to a wire-EDM finished profile",
    "manufacturing",
)
add(
    "adaptive_clearing_stepovers",
    open_curve(
        lambda t: (
            (1.0 + 0.34 * t) * np.cos(t),
            (1.0 + 0.34 * t) * np.sin(t),
        ),
        0.0, 4.4, 21,
    ),
    (0.28, 0.56, 0.84, 1.12, 1.4),
    "constant-engagement stepovers along an opening spiral clearing pass",
    "manufacturing",
)
add(
    "mold_parting_surface_clearance",
    open_curve(lambda t: (t, 0.3 * np.tanh(1.4 * (t - 2.6)) + 0.06 * t), 0.0, 5.2, 12),
    (0.1, 0.22, -0.1),
    "clearance bands about a stepped mold parting trace",
    "manufacturing",
)

# -- design and mechanical engineering -------------------------------------

add(
    "cam_follower_lift_flanks",
    open_curve(
        lambda t: (t, 0.85 * np.exp(-1.6 * (t - 2.2) ** 2)), 0.0, 4.4, 17
    ),
    (-0.14, 0.14, 0.28),
    "flank tolerance bands about a cam lift event",
    "engineering",
)
add(
    "leaf_spring_ply_stack",
    open_curve(lambda t: (t, 0.12 * t * t - 0.02 * t**3), 0.0, 4.2, 11),
    (0.14, 0.28, 0.42, 0.56),
    "ply stack lines above a tapered leaf-spring master curve",
    "engineering",
)
add(
    "bridge_arch_falsework",
    open_curve(lambda t: (t, 1.65 * np.sin(math.pi * t / 6.4)), 0.0, 6.4, 15),
    (-0.28, -0.56, -0.84),
    "falsework layers hung below a bridge arch centerline",
    "engineering",
)
add(
    "ship_hull_station_shells",
    catmull_rom(
        [(0.0, 2.1), (0.35, 1.1), (0.85, 0.4), (1.9, 0.05), (3.2, 0.0),
         (4.4, 0.12), (5.2, 0.5)],
        5, closed=False,
    ),
    (0.18, 0.36, 0.54),
    "inward plating shells from a hull station curve",
    "engineering",
)

# -- physics and CFD --------------------------------------------------------

add(
    "airfoil_camber_boundary_layers",
    open_curve(
        lambda t: (t, 0.34 * np.sin(math.pi * t / 4.4) * (1.0 - 0.16 * t / 4.4)),
        0.0, 4.4, 17,
    ),
    (0.05, 0.11, 0.19, 0.3, 0.45),
    "geometrically growing inflation layers over a camber line",
    "cfd",
)
add(
    "shock_standoff_layers",
    open_curve(lambda t: (0.9 * np.cosh(t) - 0.9, 1.35 * np.sinh(t)), -1.15, 1.15, 13),
    (0.16, 0.34, 0.55),
    "bow-shock standoff surfaces ahead of a blunt-body hyperbolic nose",
    "physics",
)
add(
    "magnetic_flux_shells",
    open_curve(lambda t: (2.1 * np.sin(t), 1.5 * np.sin(2.0 * t) * 0.5 + 1.15 * t / 2.4), 0.25, 2.4, 15),
    (0.14, 0.3, 0.48),
    "nested flux shells beside a curved field-line arc",
    "physics",
)
add(
    "cyclotron_spiral_gaps",
    open_curve(
        lambda t: ((0.35 + 0.21 * t) * np.cos(t), (0.35 + 0.21 * t) * np.sin(t)),
        0.0, 5.6, 25,
    ),
    (0.12, -0.12),
    "acceleration-gap clearance about an opening cyclotron orbit",
    "physics",
)
add(
    "optical_caustic_wavefronts",
    open_curve(lambda t: (t, 0.5 * t * t / (1.0 + 0.28 * t)), -1.9, 1.9, 13),
    (0.42, 0.84, 1.26, 1.68),
    "advancing wavefront offsets that sharpen toward a caustic cusp",
    "physics",
)

# -- space engineering ------------------------------------------------------

add(
    "nozzle_contour_liner",
    open_curve(
        lambda t: (t, 0.42 + 0.62 * np.sqrt(np.maximum(t, 0.0)) - 0.052 * t),
        0.0, 5.0, 15,
    ),
    (-0.12, -0.24),
    "ablative liner plies inside a bell-nozzle wall contour",
    "space",
)
add(
    "reentry_heatshield_standoff",
    open_curve(lambda t: (2.6 * np.sin(t), 1.5 * (np.cos(t) - 1.0)), -1.05, 1.05, 13),
    (0.2, 0.42, 0.66),
    "standoff insulation gaps above a blunt reentry heat-shield arc",
    "space",
)
add(
    "launch_gantry_swing_clearance",
    catmull_rom(
        [(0.0, 0.0), (0.7, 1.2), (1.05, 2.5), (1.6, 3.6), (2.7, 4.3),
         (4.1, 4.55)],
        5, closed=False,
    ),
    (-0.3, -0.6),
    "swing-arm retraction clearance beside a launch gantry path",
    "space",
)
add(
    "orbital_transfer_corridor",
    open_curve(
        lambda t: ((3.4 - 0.5 * t) * np.cos(t), (2.5 - 0.32 * t) * np.sin(t)),
        0.15, 2.75, 17,
    ),
    (0.22, -0.22),
    "navigation corridor about an elliptic transfer arc segment",
    "space",
)

# -- logistics, civil and geo ----------------------------------------------

add(
    "agv_aisle_corridor",
    catmull_rom(
        [(0.0, 0.0), (1.4, 0.15), (2.6, 0.9), (3.9, 0.95), (5.2, 0.3),
         (6.6, 0.4)],
        5, closed=False,
    ),
    (-0.4, 0.4),
    "drivable corridor edges about an AGV guide path",
    "logistics",
)
add(
    "highway_lane_offsets",
    meander(9.0, lambda x: 1.15 * np.sin(0.75 * x) * np.exp(-0.045 * x), 21),
    (-0.75, -0.375, 0.375, 0.75),
    "two travel lanes each side of a highway centerline",
    "logistics",
)
add(
    "pipeline_right_of_way",
    seeded_walk(20260811, 24, step=0.85, turn_scale=0.24),
    (0.55, 1.1, -0.55, -1.1),
    "construction and permanent right-of-way strips along a pipeline route",
    "logistics",
)
add(
    "river_flood_setbacks",
    meander(10.0, lambda x: 1.35 * np.sin(0.9 * x + 0.6) + 0.3 * np.sin(2.1 * x), 27),
    (0.5, 1.0, 1.55),
    "graded flood setback lines along a meandering river bank",
    "geoscience",
)
add(
    "coastal_buffer_zones",
    seeded_walk(31, 26, step=0.8, turn_scale=0.5),
    (0.5, 1.05, 1.65),
    "protection buffers seaward of an irregular coastline trace",
    "geoscience",
)
add(
    "glacier_retreat_isochrones",
    seeded_walk(97, 20, step=0.9, turn_scale=0.38),
    (-0.45, -0.95, -1.5, -2.1),
    "retreat isochrones behind a mapped glacier terminus",
    "geoscience",
)
add(
    "ski_piste_grooming_passes",
    open_curve(
        lambda t: (t, 0.85 * np.cos(0.9 * t) + 0.24 * np.cos(2.3 * t)),
        0.0, 7.5, 19,
    ),
    (0.4, 0.8, 1.2),
    "parallel grooming passes across a piste fall-line trace",
    "logistics",
)

# -- microscale and electronics --------------------------------------------

add(
    "microfluidic_channel_walls",
    meander(6.0, 0.62, 25, frequency=2.6),
    (-0.14, 0.14),
    "wall pair of a serpentine microfluidic mixing channel",
    "manufacturing",
)
add(
    "pcb_trace_keepouts",
    catmull_rom(
        [(0.0, 0.0), (0.9, 0.05), (1.5, 0.7), (2.5, 0.75), (3.3, 0.2),
         (4.3, 0.25), (5.0, 0.85)],
        5, closed=False,
    ),
    (0.11, 0.24, -0.11, -0.24),
    "clearance and keepout bands about a routed differential trace",
    "manufacturing",
)

# -- abstract ---------------------------------------------------------------

add(
    "calligraphy_stroke_weights",
    catmull_rom(
        [(0.0, 0.0), (0.75, 0.95), (1.9, 1.15), (2.6, 0.35), (3.5, 0.1),
         (4.5, 0.9), (5.05, 1.9)],
        6, closed=False,
    ),
    (0.09, 0.2, -0.09, -0.2),
    "nib-weight envelopes about a calligraphic flourish",
    "abstract",
)
add(
    "archimedean_cusp_ripples",
    open_curve(lambda t: (2.05 * np.cos(t), 1.35 * np.sin(t)), 0.25, 2.9, 15),
    (0.7, 1.35, 2.05),
    "inward ripples pushed past the curvature radius: cusps kept, never trimmed",
    "abstract",
)


def build(case: OffsetCase) -> CubicPHSplineOpen:
    return CubicPHSplineOpen(case.points)


if __name__ == "__main__":
    run(CASES, build, OUT, "open cubic PH source")
