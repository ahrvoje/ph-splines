"""Render 32 closed cubic-PH exact-offset studies into this folder.

Run from the repository root:
    python examples/cubic_closed_offset/generate_examples.py

Use ``--check-only`` to construct and verify every offset without drawing.
Each case builds a :class:`CubicPHSplineClosed`, requests several exact
rational NURBS offsets, and verifies them against ``r(u) + d * N_L(u)``
before publishing an image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from offset_common import (  # noqa: E402
    OffsetCase,
    catmull_rom,
    periodic,
    polar,
    run,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ph_spline import CubicPHSplineClosed  # noqa: E402

OUT = Path(__file__).resolve().parent
CASES: list[OffsetCase] = []


def add(name, points, distances, note, category, **kwargs):
    CASES.append(OffsetCase(name, points, tuple(distances), note, category,
                            **kwargs))


def smooth_square(values: np.ndarray, sharpness: float = 0.1) -> np.ndarray:
    return values / np.sqrt(values * values + sharpness * sharpness)


# -- manufacturing ----------------------------------------------------------

add(
    "pocket_milling_stepovers",
    catmull_rom(
        [(-2.6, -1.1), (-1.2, -1.55), (0.9, -1.35), (2.5, -1.5), (3.0, -0.2),
         (2.1, 1.15), (0.3, 1.5), (-1.8, 1.3), (-2.9, 0.3)],
        5,
    ),
    (0.28, 0.56, 0.84, 1.12),
    "inward roughing stepovers filling a milled pocket boundary",
    "manufacturing",
    fill=True,
)
add(
    "gasket_seal_beads",
    periodic(
        lambda t: (
            2.25 * np.cos(t) + 0.28 * np.cos(2.0 * t),
            1.35 * np.sin(t) + 0.14 * np.sin(3.0 * t),
        ),
        61,
    ),
    (-0.16, 0.16, 0.32),
    "sealing bead lands each side of a stamped gasket centerline",
    "manufacturing",
)
add(
    "stamping_die_binder_ring",
    polar(lambda t: 1.9 + 0.22 * np.cos(3.0 * t) + 0.06 * np.cos(6.0 * t), 71),
    (-0.24, -0.5),
    "binder ring lands wrapped around a stamping die opening",
    "manufacturing",
)
add(
    "mold_shrink_compensation",
    periodic(
        lambda t: (
            2.05 * np.cos(t) - 0.34 * np.cos(2.0 * t),
            1.4 * np.sin(t) + 0.1 * np.sin(2.0 * t),
        ),
        53,
    ),
    (-0.12, -0.25),
    "shrink-compensated cavity lines grown from a nominal molding contour",
    "manufacturing",
)
add(
    "casting_riser_neck_rings",
    polar(lambda t: 1.35 + 0.16 * np.cos(2.0 * t - 0.5) + 0.05 * np.cos(5.0 * t), 47),
    (-0.18, -0.38, -0.6),
    "riser neck transition rings around a casting boss outline",
    "manufacturing",
)
add(
    "forging_flash_land",
    periodic(
        lambda t: (
            1.85 * np.cos(t) + 0.2 * np.cos(3.0 * t),
            1.2 * np.sin(t) - 0.16 * np.sin(3.0 * t),
        ),
        57,
    ),
    (0.14, 0.3, -0.14),
    "flash land and gutter bands about a forging die parting contour",
    "manufacturing",
)
add(
    "wire_race_bearing_groove",
    polar(lambda t: 1.75 + 0.045 * np.cos(9.0 * t), 91),
    (-0.12, 0.12),
    "raceway groove walls about a wavy wire-race bearing centerline",
    "manufacturing",
)
add(
    "gear_blank_machining_stock",
    polar(
        lambda t: 1.65
        * (1.0 + 0.045 * smooth_square(np.sin(11.0 * t), 0.14)),
        143,
    ),
    (-0.16, -0.34),
    "machining stock envelopes outside a smoothed gear blank",
    "manufacturing",
)

# -- design and mechanical engineering -------------------------------------

add(
    "cam_ring_clearance",
    polar(lambda t: 1.55 + 0.34 * np.cos(t + 0.4) + 0.1 * np.cos(2.0 * t), 49),
    (-0.2, 0.2),
    "follower clearance bands about an internal cam ring profile",
    "engineering",
)
add(
    "impeller_shroud_clearance",
    periodic(
        lambda t: (
            1.95 * np.cos(t) + 0.24 * np.cos(4.0 * t),
            1.95 * np.sin(t) - 0.24 * np.sin(4.0 * t) * 0.6,
        ),
        73,
    ),
    (-0.14, -0.3),
    "running-clearance shells outside a shrouded impeller rim",
    "engineering",
)
add(
    "harmonic_drive_flexspline",
    polar(lambda t: 1.7 + 0.085 * np.cos(2.0 * t), 61),
    (-0.11, 0.11, 0.22),
    "tooth-land offsets about an elliptic flexspline neutral line",
    "engineering",
)
add(
    "labyrinth_seal_teeth",
    polar(lambda t: 1.5 + 0.05 * np.cos(6.0 * t + 0.8), 79),
    (-0.13, -0.27, -0.42, -0.58),
    "successive labyrinth seal teeth grown from a rotor land",
    "engineering",
)
add(
    "turbofan_casing_treatment",
    periodic(
        lambda t: (
            2.15 * np.cos(t) + 0.1 * np.cos(5.0 * t),
            2.15 * np.sin(t) + 0.1 * np.sin(5.0 * t),
        ),
        81,
    ),
    (-0.18, -0.4),
    "casing treatment grooves outside a fan flow-path circle",
    "engineering",
)
add(
    "guitar_body_binding",
    catmull_rom(
        [(0.0, 1.95), (1.15, 1.6), (1.5, 0.7), (1.25, -0.2), (1.65, -1.15),
         (1.05, -2.05), (0.0, -2.3), (-1.05, -2.05), (-1.65, -1.15),
         (-1.25, -0.2), (-1.5, 0.7), (-1.15, 1.6)],
        5,
    ),
    (-0.14, -0.3),
    "binding and purfling channels routed inside a guitar body outline",
    "design",
    fill=True,
)
add(
    "picture_frame_moulding",
    periodic(
        lambda t: (
            2.5 * smooth_square(np.cos(t), 0.33),
            1.8 * smooth_square(np.sin(t), 0.33),
        ),
        77,
    ),
    (-0.2, -0.42, -0.66),
    "stepped moulding profiles wrapped around a rounded frame",
    "design",
)
add(
    "medallion_relief_bands",
    polar(lambda t: 1.75 + 0.16 * np.cos(7.0 * t), 105),
    (0.14, 0.3, 0.48),
    "die-relief bands sinking into a seven-lobe medallion rim",
    "design",
)

# -- sport and mobility -----------------------------------------------------

add(
    "race_track_edges",
    catmull_rom(
        [(-2.8, -0.3), (-1.9, -1.2), (-0.4, -1.35), (0.7, -0.75),
         (1.7, -1.15), (2.8, -0.35), (2.45, 0.75), (1.45, 0.62),
         (0.8, 1.35), (-0.25, 1.1), (-0.9, 0.45), (-1.85, 0.85),
         (-2.75, 0.45)],
        5,
    ),
    (-0.22, 0.22),
    "track edges each side of a grand-prix racing centerline",
    "mobility",
)
add(
    "stadium_running_lanes",
    periodic(
        lambda t: (
            2.6 * smooth_square(np.cos(t), 0.62),
            1.35 * np.sin(t),
        ),
        69,
    ),
    (-0.22, -0.44, -0.66, -0.88),
    "four stadium lanes measured outward from the inner kerb line",
    "mobility",
)
add(
    "velodrome_sprint_lines",
    periodic(lambda t: (2.75 * np.cos(t), 1.55 * np.sin(t)), 45),
    (-0.18, -0.4, -0.85),
    "sprinter, stayer and blue-band lines above the velodrome datum",
    "mobility",
)
add(
    "agv_loop_lanes",
    catmull_rom(
        [(-2.5, -1.2), (-0.6, -1.6), (1.7, -1.35), (2.7, -0.3), (2.2, 0.95),
         (0.4, 1.5), (-1.7, 1.35), (-2.85, 0.25)],
        5,
    ),
    (-0.3, 0.3),
    "bidirectional AGV lanes about a warehouse loop spine",
    "logistics",
)
add(
    "robot_safety_fence",
    catmull_rom(
        [(-2.2, -1.5), (0.2, -1.9), (2.35, -1.25), (2.9, 0.35), (1.7, 1.55),
         (-0.7, 1.8), (-2.75, 0.9)],
        5,
    ),
    (-0.35, -0.75, -1.2),
    "graded safety fences outside a robot workcell envelope",
    "logistics",
    fill=True,
)

# -- physics and space ------------------------------------------------------

add(
    "tokamak_first_wall",
    periodic(
        lambda t: (
            1.05 * np.cos(t + 0.30 * np.sin(t)),
            1.34 * np.sin(t),
        ),
        67,
    ),
    (-0.14, -0.3, -0.48),
    "first-wall and blanket standoffs outside a D-shaped flux surface",
    "physics",
)
add(
    "magnet_pole_shoe_gap",
    periodic(
        lambda t: (
            1.9 * np.cos(t) + 0.22 * np.cos(2.0 * t),
            1.35 * np.sin(t),
        ),
        49,
    ),
    (-0.15, 0.15),
    "air-gap faces about a shaped magnet pole-shoe midline",
    "physics",
)
add(
    "storage_ring_aperture",
    polar(lambda t: 2.1 + 0.14 * np.cos(4.0 * t + 0.6), 73),
    (0.2, 0.42),
    "vacuum aperture reductions inside a storage-ring reference orbit",
    "physics",
)
add(
    "cryostat_mli_blankets",
    periodic(
        lambda t: (
            1.75 * np.cos(t) - 0.2 * np.cos(2.0 * t),
            1.55 * np.sin(t) + 0.12 * np.sin(2.0 * t),
        ),
        51,
    ),
    (-0.12, -0.26, -0.42, -0.6),
    "multi-layer insulation blankets wrapped over a cryostat shell section",
    "space",
)
add(
    "satellite_keepout_rings",
    polar(lambda t: 1.45 + 0.19 * np.cos(3.0 * t - 0.7), 55),
    (-0.3, -0.65, -1.05),
    "attitude-thruster plume keepout rings around a bus cross-section",
    "space",
)
add(
    "lunar_berm_terracing",
    polar(lambda t: 1.8 + 0.24 * np.cos(2.0 * t) + 0.1 * np.cos(5.0 * t - 1.1), 63),
    (-0.28, -0.6, -0.95),
    "regolith berm terraces graded outward from a habitat perimeter",
    "space",
)

# -- geo and environment ----------------------------------------------------

add(
    "island_territorial_rings",
    polar(
        lambda t: 1.5
        + 0.28 * np.cos(3.0 * t + 0.4)
        + 0.12 * np.cos(7.0 * t + 1.9)
        + 0.05 * np.cos(11.0 * t),
        95,
    ),
    (-0.4, -0.9, -1.5),
    "smoothing legal-distance rings around a jagged island shoreline",
    "geoscience",
)
add(
    "atoll_lagoon_buffers",
    periodic(
        lambda t: (
            2.3 * np.cos(t) + 0.2 * np.cos(2.0 * t - 0.6),
            1.5 * np.sin(t) + 0.16 * np.sin(3.0 * t),
        ),
        59,
    ),
    (0.22, 0.48, 0.78),
    "lagoon-side ecological buffers inside an atoll reef crest",
    "geoscience",
)
add(
    "reservoir_drawdown_lines",
    catmull_rom(
        [(-2.4, -0.9), (-0.9, -1.55), (1.1, -1.2), (2.6, -1.45), (3.05, 0.0),
         (1.9, 1.2), (0.1, 1.6), (-1.9, 1.25), (-2.9, 0.2)],
        5,
    ),
    (0.26, 0.55, 0.9),
    "seasonal drawdown shorelines nested inside a reservoir outline",
    "geoscience",
    fill=True,
)

# -- abstract ---------------------------------------------------------------

add(
    "superellipse_tolerance_bands",
    periodic(
        lambda t: (
            2.2 * smooth_square(np.cos(t), 0.45),
            1.6 * smooth_square(np.sin(t), 0.45),
        ),
        61,
    ),
    (-0.16, 0.16, 0.34),
    "symmetric tolerance bands about a superelliptic master gauge",
    "abstract",
)
add(
    "star_cusp_cascade",
    polar(lambda t: 1.9 + 0.3 * np.cos(5.0 * t), 85),
    (0.5, 0.95, 1.45),
    "inward offsets pushed past the curvature radius: cusps and loops kept",
    "abstract",
)


def build(case: OffsetCase) -> CubicPHSplineClosed:
    return CubicPHSplineClosed(case.points)


if __name__ == "__main__":
    run(CASES, build, OUT, "closed cubic PH source")
