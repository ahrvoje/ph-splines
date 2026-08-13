"""Render 32 open PH B-spline offset distance-query studies here.

Run from the repository root:
    python examples/bspline_offset_distance/generate_examples.py

Use ``--check-only`` to verify every offset and its distance API without
drawing.  Each case builds a :class:`PHBSplineOpen` (several at raised
continuity order, so the rational offsets range up to degree 21), compiles
the exact offset NURBS with its verified metric certificate, and places all
distance stations with ``point_at_length`` along the offset locus.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from offset_distance_common import (  # noqa: E402
    DistanceCase,
    catmull_rom,
    meander,
    open_curve,
    run,
    seeded_walk,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ph_spline import PHBSplineOpen  # noqa: E402

OUT = Path(__file__).resolve().parent
CASES: list[DistanceCase] = []


def add(name, points, distance, note, category, **kwargs):
    CASES.append(DistanceCase(name, points, distance, note, category, **kwargs))


def _clothoid(scale: float, count: int) -> list[list[float]]:
    ts = np.linspace(0.0, 1.0, count)
    xs, ys = [], []
    for t in ts:
        u = np.linspace(0.0, t, 220)
        xs.append(scale * np.trapezoid(np.cos(2.2 * u * u), u))
        ys.append(scale * np.trapezoid(np.sin(2.2 * u * u), u))
    return np.column_stack((xs, ys)).tolist()


# -- semiconductor and displays ----------------------------------------------

add(
    "oled_dispense_serpentine",
    meander(7.6, lambda x: 0.5 * np.sin(1.5 * x) * (1 - 0.06 * x), 16),
    0.16,
    "encapsulation dispense edge; drop volume scheduled by bead travel",
    "displays",
    stations=33,
)
add(
    "wire_bond_capillary_loop",
    open_curve(lambda t: (2.4 * t, 0.9 * np.sin(math.pi * t) * (1 - 0.3 * t)),
               0.0, 1.0, 12),
    0.12,
    "capillary clearance path over a die edge; loop checks by path travel",
    "semiconductor",
    stations=21,
)
add(
    "euv_mirror_meridian_polish",
    open_curve(lambda t: (2.8 * t, 0.35 * t * t), 0.0, 1.2, 13),
    -0.14,
    "ion-beam figuring meridian; dwell-time bins by true meridian travel",
    "semiconductor",
    stations=25, kwargs={"g_order": 4}, highlight=(0.35, 0.5),
)
add(
    "probe_needle_cleaning_serpentine",
    meander(5.4, 0.42, 14, frequency=2.1),
    0.11,
    "abrasive pad track; refresh points at constant pad travel",
    "semiconductor",
    stations=27,
)

# -- aerospace and space -----------------------------------------------------

add(
    "winglet_blend_bond_line",
    catmull_rom(
        [(0.0, 0.0), (1.3, 0.35), (2.6, 0.5), (3.8, 1.05), (4.6, 2.0)],
        6, closed=False,
    ),
    0.15,
    "winglet blend bond line; fastener pitch by true bond-line length",
    "aerospace",
    stations=27,
)
add(
    "satellite_harness_routing",
    seeded_walk(2218, 14, step=0.9, turn_scale=0.28),
    0.18,
    "harness keep-out beside a panel route; tie-downs by harness travel",
    "space",
    stations=27, highlight=(0.5, 0.68),
)
add(
    "reusable_booster_weld_land",
    open_curve(lambda t: (t, 0.28 * np.sin(1.9 * t) + 0.1 * t), 0.0, 6.2, 15),
    -0.17,
    "friction-stir weld land; pin-tool checkpoints by land travel",
    "space",
    stations=29, kwargs={"g_order": 4},
)
add(
    "uav_swarm_corridor_edge",
    meander(8.8, lambda x: 0.75 * np.sin(0.9 * x) * np.exp(-0.04 * x), 17),
    0.4,
    "traffic-corridor edge; conflict gates at equal corridor chainage",
    "aerospace",
    stations=35,
)

# -- manufacturing ------------------------------------------------------------

add(
    "afp_steered_tow_high_order",
    open_curve(lambda t: (t, 0.62 * np.sin(0.72 * t)), 0.0, 8.2, 15),
    0.24,
    "G4 steered tow with one tow-width offset; roller force by tow travel",
    "manufacturing",
    stations=31, extra=(0.48,), kwargs={"g_order": 4},
)
add(
    "waterjet_glass_scribe",
    catmull_rom(
        [(0.0, 0.0), (1.1, 0.75), (2.5, 0.55), (3.6, 1.2), (5.0, 0.85),
         (6.1, 1.5)],
        6, closed=False,
    ),
    0.1,
    "scribe wall clearance in chemically strengthened glass",
    "manufacturing",
    stations=29,
)
add(
    "robot_deburr_edge_pass",
    seeded_walk(7731, 13, step=0.95, turn_scale=0.31),
    -0.2,
    "compliant deburr pass; media replacement by cumulative edge travel",
    "manufacturing",
    stations=25,
)
add(
    "additive_daed_bead_path",
    meander(6.4, lambda x: 0.4 * np.sin(1.25 * x) + 0.07 * x, 15),
    0.19,
    "directed-energy bead edge; melt-pool cameras by bead travel",
    "manufacturing",
    stations=27, highlight=(0.15, 0.3),
)
add(
    "press_brake_air_bend_line",
    open_curve(lambda t: (3.5 * t, 0.7 * t * t * (3 - 2 * t)), 0.0, 1.0, 11),
    0.13,
    "die-shoulder clearance beside an air-bend line",
    "manufacturing",
    stations=21,
)

# -- robotics and logistics ---------------------------------------------------

add(
    "agv_cross_dock_weave",
    meander(9.2, 0.95, 18, frequency=2.6),
    0.42,
    "cross-dock weave lane edge; RFID gates by lane chainage",
    "logistics",
    stations=37,
)
add(
    "exoskeleton_gait_guide",
    open_curve(lambda t: (2.2 * t, 0.5 * np.sin(2.4 * math.pi * t) * (0.4 + 0.6 * t)),
               0.0, 1.0, 14),
    0.16,
    "hip-trajectory guide rail; assist-torque stages by rail travel",
    "robotics",
    stations=25,
)
add(
    "surgical_robot_port_approach",
    seeded_walk(6106, 12, step=0.7, turn_scale=0.27),
    -0.12,
    "trocar approach margin; haptic zones at equal approach travel",
    "medical-robotics",
    stations=21, highlight=(0.68, 0.9),
)
add(
    "warehouse_picker_racetrack",
    catmull_rom(
        [(0.0, 0.0), (1.7, 0.4), (3.2, 0.1), (4.8, 0.7), (6.4, 0.35),
         (7.8, 0.9)],
        6, closed=False,
    ),
    0.3,
    "pick-arm reach envelope beside an aisle line",
    "logistics",
    stations=31,
)

# -- energy -------------------------------------------------------------------

add(
    "hydrogen_pipe_bend_corridor",
    open_curve(lambda t: (2.9 * np.sin(t), 1.9 * (1 - np.cos(t))), 0.1, 1.5, 12),
    0.26,
    "blast setback along a hydrogen header bend; sensors by pipe travel",
    "energy",
    stations=23,
)
add(
    "solar_tracker_cable_droop",
    open_curve(lambda t: (t, 0.42 * (np.cosh(t - 2.4) - 1.0)), 0.0, 4.8, 14),
    -0.24,
    "cable-tray droop clearance; cleats at constant tray travel",
    "energy",
    stations=27,
)
add(
    "geothermal_casing_dogleg",
    seeded_walk(9414, 15, step=1.0, turn_scale=0.16, drift=0.03),
    0.35,
    "casing wall standoff through a dogleg; centralizers by measured depth",
    "energy",
    stations=29, kwargs={"g_order": 4},
)
add(
    "tidal_blade_leading_edge",
    open_curve(lambda t: (3.6 * t, 0.75 * np.sqrt(np.clip(t, 1e-9, None)) * (1 - 0.55 * t)),
               0.02, 1.0, 13),
    0.11,
    "leading-edge cavitation strip; erosion coupons by strip travel",
    "energy",
    stations=23,
)

# -- transport ----------------------------------------------------------------

add(
    "hyperloop_transition_spiral",
    _clothoid(4.4, 15),
    0.2,
    "clothoid guideway easement; jerk gates at equal guideway travel",
    "transport",
    stations=27, kwargs={"g_order": 4}, highlight=(0.55, 0.75),
)
add(
    "harbor_dredge_channel_edge",
    catmull_rom(
        [(0.0, 0.0), (1.8, 0.5), (3.4, 0.3), (5.2, 1.1), (7.0, 0.8),
         (8.4, 1.6)],
        6, closed=False,
    ),
    0.5,
    "dredged channel toe line; survey cross-sections by channel chainage",
    "transport",
    stations=33,
)
add(
    "funicular_track_easement",
    open_curve(lambda t: (2.8 * t, 1.3 * t * t), 0.0, 1.3, 12),
    -0.19,
    "rack-rail clearance easement on a steep grade",
    "transport",
    stations=23,
)

# -- environment and science --------------------------------------------------

add(
    "river_levee_setback",
    seeded_walk(1580, 17, step=1.1, turn_scale=0.2),
    0.55,
    "levee toe setback along a meander; borings by levee chainage",
    "environment",
    stations=33,
)
add(
    "glacier_crevasse_survey",
    meander(8.0, lambda x: 0.6 * np.sin(0.85 * x) + 0.12 * np.sin(2.7 * x), 17),
    0.3,
    "safe-standoff survey track; GPR shots at constant track travel",
    "science",
    stations=31,
)
add(
    "coastal_buffer_line",
    catmull_rom(
        [(0.0, 0.2), (1.5, 0.7), (2.9, 0.4), (4.3, 1.3), (5.9, 1.0),
         (7.2, 1.8), (8.5, 1.5)],
        6, closed=False,
    ),
    0.45,
    "erosion buffer inland of a shoreline trace; monuments by chainage",
    "environment",
    stations=35, fill=False,
)
add(
    "seismic_streamer_feather",
    open_curve(lambda t: (t, 0.24 * np.sin(0.8 * t) + 0.05 * t), 0.0, 9.0, 18),
    -0.35,
    "streamer feather envelope; hydrophone groups by cable distance",
    "science",
    stations=37,
)

add(
    "metro_screen_door_line",
    open_curve(lambda t: (4.0 * t, 0.35 * np.sin(1.6 * math.pi * t)), 0.0, 1.0, 13),
    0.22,
    "platform screen-door line; door leaves at equal platform travel",
    "transport",
    stations=25,
)
add(
    "laser_bar_facet_scan",
    meander(5.0, lambda x: 0.3 * np.sin(1.8 * x) * (1 - 0.08 * x), 13),
    -0.1,
    "facet photoluminescence scan lane; exposure bins by lane travel",
    "photonics",
    stations=23,
)

# -- high order and verification ---------------------------------------------

add(
    "g6_optical_bench_flexure",
    open_curve(lambda t: (2.6 * t, 0.45 * np.sin(math.pi * t)), 0.0, 1.0, 10),
    0.14,
    "G6 flexure blade edge (degree-21 rational offset); probes by edge arc",
    "verification",
    stations=21, kwargs={"g_order": 6},
)
add(
    "high_order_cusp_sweep",
    open_curve(lambda t: (2.2 * np.cos(t), 1.4 * np.sin(t)), 0.3, 2.85, 15),
    1.05,
    "beyond-critical inward sweep at G3: certified cusps under high degree",
    "verification",
    stations=39,
)


def build(case: DistanceCase) -> PHBSplineOpen:
    return PHBSplineOpen(case.points, **case.kwargs)


if __name__ == "__main__":
    run(CASES, build, OUT, "open PH B-spline source")
