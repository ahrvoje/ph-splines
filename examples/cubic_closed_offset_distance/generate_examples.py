"""Render 32 closed cubic-PH offset distance-query studies into this folder.

Run from the repository root:
    python examples/cubic_closed_offset_distance/generate_examples.py

Use ``--check-only`` to verify every offset and its distance API without
drawing.  Each case builds a :class:`CubicPHSplineClosed` ring, compiles an
exact rational NURBS offset with its verified metric certificate, and lays
out distance stations with ``point_at_length`` along the offset loop - the
one-traversal distance domain ``[0, L]`` of the specification, with no
wrap-around.
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
    periodic,
    polar,
    run,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ph_spline import CubicPHSplineClosed  # noqa: E402

OUT = Path(__file__).resolve().parent
CASES: list[DistanceCase] = []


def add(name, points, distance, note, category, **kwargs):
    CASES.append(DistanceCase(name, points, distance, note, category, **kwargs))


# -- turbomachinery and propulsion -------------------------------------------

add(
    "compressor_blisk_shroud_ring",
    polar(lambda t: 2.0 + 0.16 * np.cos(9 * t), 36),
    0.2,
    "shroud clearance ring; blade-tip timing probes by rim travel",
    "turbomachinery",
    stations=37, fill=True,
)
add(
    "turbine_disc_firtree_pitchline",
    polar(lambda t: 1.8 + 0.1 * np.cos(12 * t), 40),
    -0.14,
    "broach pitch line inside the firtree row; slots by pitch-arc length",
    "turbomachinery",
    stations=41,
)
add(
    "impeller_shroud_wear_ring",
    polar(lambda t: 1.6 + 0.28 * np.sin(2 * t) * np.cos(t), 26),
    0.22,
    "wear-ring land around a pump shroud; labyrinth teeth by land travel",
    "turbomachinery",
    stations=29, highlight=(0.25, 0.4),
)
add(
    "rocket_injector_faceplate_ring",
    polar(lambda t: 1.5 + 0.07 * np.cos(6 * t), 30),
    -0.18,
    "coaxial-element mounting ring; injector posts at equal ring travel",
    "propulsion",
    stations=31,
)

# -- e-mobility and machines --------------------------------------------------

add(
    "pm_rotor_magnet_retention",
    polar(lambda t: 1.7 + 0.12 * np.cos(8 * t), 32),
    0.15,
    "retention-sleeve interface; balance-correction sites by rim length",
    "e-mobility",
    stations=33, fill=True,
)
add(
    "stator_slot_liner_loop",
    polar(lambda t: 1.9 + 0.18 * np.cos(12 * t), 48),
    -0.12,
    "slot-liner insertion loop; creepage checks at constant liner travel",
    "e-mobility",
    stations=49,
)
add(
    "inductive_charge_coil_shield",
    polar(lambda t: 1.8 + 0.35 * np.cos(4 * t) * 0.5, 24),
    0.28,
    "ferrite shield rim beyond the coil; flux probes by rim distance",
    "e-mobility",
    stations=25, highlight=(0.55, 0.7),
)
add(
    "battery_cooling_ribbon_loop",
    catmull_rom(
        [(2.1, 0.0), (1.3, 1.5), (-0.6, 1.9), (-2.0, 0.8), (-1.9, -0.9),
         (-0.3, -1.8), (1.4, -1.4)],
        5,
    ),
    0.17,
    "serpentine ribbon return loop; thermocouples by coolant-wall travel",
    "e-mobility",
    stations=31,
)

# -- fusion and big science ---------------------------------------------------

add(
    "tokamak_first_wall_standoff",
    periodic(lambda t: (2.4 + 0.9 * np.cos(t + 0.28 * np.sin(t)),
                        1.5 * np.sin(t)), 28),
    0.24,
    "plasma-facing standoff of a D-shaped section; tiles by wall-arc pitch",
    "fusion",
    stations=33, fill=True,
)
add(
    "cryostat_thermal_shield_ring",
    polar(lambda t: 2.3 + 0.05 * np.cos(3 * t), 22),
    -0.3,
    "MLI shield ring inside the cryostat; strap anchors by ring travel",
    "fusion",
    stations=25,
)
add(
    "synchrotron_vacuum_chamber",
    periodic(lambda t: (2.6 * np.cos(t), 1.15 * np.sin(t)), 24),
    -0.2,
    "racetrack-like chamber wall; BPM buttons at equal beam-wall travel",
    "accelerators",
    stations=29,
)
add(
    "undulator_magnet_girder_rim",
    polar(lambda t: 2.0 + 0.09 * np.cos(2 * t) + 0.05 * np.sin(5 * t), 26),
    0.16,
    "girder alignment rim; fiducial nests at constant rim distance",
    "accelerators",
    stations=27, highlight=(0.1, 0.22),
)

# -- sealing, bearings, and precision mechanics -------------------------------

add(
    "o_ring_gland_centerline",
    polar(lambda t: 1.6 + 0.24 * np.cos(2 * t), 20),
    0.13,
    "gland outer wall about the seal centerline; volume checks by travel",
    "sealing",
    stations=21,
)
add(
    "hydrostatic_bearing_land",
    polar(lambda t: 1.85 + 0.06 * np.cos(4 * t), 24),
    -0.16,
    "bearing land inside the pad ring; feed orifices by land travel",
    "precision-mechanics",
    stations=25,
)
add(
    "harmonic_drive_flexspline_wall",
    polar(lambda t: 1.75 + 0.08 * np.cos(2 * t), 28),
    0.1,
    "flexspline wall offset; strain gauges at equal wall-arc positions",
    "precision-mechanics",
    stations=29,
)
add(
    "ball_screw_return_channel",
    catmull_rom(
        [(1.9, 0.2), (0.9, 1.6), (-0.8, 1.7), (-1.9, 0.5), (-1.5, -1.1),
         (0.2, -1.75), (1.5, -1.2)],
        5,
    ),
    -0.15,
    "ball return-channel wall; witness marks by channel travel",
    "precision-mechanics",
    stations=27,
)

# -- automotive and motorsport ------------------------------------------------

add(
    "race_track_drs_zones",
    catmull_rom(
        [(2.6, 0.0), (1.9, 1.5), (0.0, 2.1), (-2.0, 1.4), (-2.6, -0.3),
         (-1.2, -1.8), (0.9, -2.0), (2.2, -1.1)],
        5,
    ),
    0.3,
    "outer track edge; marshal posts and DRS gates by true edge chainage",
    "motorsport",
    stations=41, fill=True, highlight=(0.62, 0.78),
)
add(
    "cam_lobe_follower_envelope",
    polar(lambda t: 1.35 + 0.42 * np.exp(-6.0 * np.sin(0.5 * t) ** 2), 30),
    0.18,
    "roller-follower envelope of a cam lobe; lift audits by envelope arc",
    "automotive",
    stations=31,
)
add(
    "tire_mold_tread_ring",
    polar(lambda t: 2.1 + 0.07 * np.cos(18 * t), 54),
    0.12,
    "tread-ring parting surface; sipe blades pitched by ring travel",
    "automotive",
    stations=55,
)
add(
    "ev_reduction_gear_pocket",
    polar(lambda t: 1.65 + 0.1 * np.cos(6 * t), 30),
    -0.2,
    "lubrication pocket wall inside the gear blank; jets by wall travel",
    "automotive",
    stations=27,
)

# -- medical implants ---------------------------------------------------------

add(
    "acetabular_cup_rim_reamer",
    periodic(lambda t: (1.9 * np.cos(t), 1.55 * np.sin(t) + 0.14 * np.sin(2 * t)), 22),
    -0.17,
    "reamer clearance inside the cup rim; osteotome stops by rim travel",
    "medical",
    stations=23,
)
add(
    "annuloplasty_ring_suture_line",
    periodic(lambda t: (1.75 * np.cos(t) + 0.25 * np.cos(2 * t),
                        1.25 * np.sin(t)), 20),
    0.14,
    "suture line outside a saddle-projected annulus; bites by ring length",
    "medical",
    stations=27, highlight=(0.3, 0.45),
)
add(
    "cranial_plate_boundary",
    catmull_rom(
        [(1.8, 0.3), (0.9, 1.5), (-0.6, 1.6), (-1.7, 0.6), (-1.6, -0.8),
         (-0.2, -1.5), (1.2, -1.2)],
        5,
    ),
    0.12,
    "patient-specific plate boundary; screw holes by boundary travel",
    "medical",
    stations=29,
)

# -- energy and infrastructure ------------------------------------------------

add(
    "wind_tower_flange_bolt_circle",
    polar(lambda t: 2.15 + 0.04 * np.cos(2 * t), 18),
    -0.22,
    "flange inner face from the shell line; studs at equal face travel",
    "energy",
    stations=37,
)
add(
    "pressure_vessel_manway_reinforce",
    periodic(lambda t: (1.55 * np.cos(t), 1.15 * np.sin(t)), 16),
    0.26,
    "reinforcement pad boundary; weld inspection by pad-edge travel",
    "energy",
    stations=25,
)
add(
    "penstock_stiffener_ring",
    polar(lambda t: 2.0 + 0.06 * np.sin(3 * t), 20),
    0.2,
    "external stiffener ring; anode blocks at constant ring distance",
    "energy",
    stations=27,
)
add(
    "containment_berm_setback",
    catmull_rom(
        [(2.4, 0.2), (1.4, 1.7), (-0.4, 2.0), (-2.2, 1.2), (-2.5, -0.5),
         (-1.0, -1.9), (1.0, -2.1), (2.3, -1.0)],
        5,
    ),
    0.35,
    "regulatory setback around a tank berm; markers by fence chainage",
    "infrastructure",
    stations=33, fill=True,
)

# -- consumer and architecture ------------------------------------------------

add(
    "smartwatch_bezel_antenna",
    polar(lambda t: 1.5 + 0.05 * np.cos(4 * t), 20),
    0.11,
    "bezel antenna keep-out ring; feed points at equal ring travel",
    "consumer",
    stations=21,
)
add(
    "speaker_surround_roll",
    polar(lambda t: 1.7 + 0.12 * np.cos(2 * t) - 0.05 * np.cos(6 * t), 24),
    0.15,
    "surround roll glue line; compliance probes by glue-line travel",
    "consumer",
    stations=25,
)
add(
    "stadium_roof_compression_ring",
    periodic(lambda t: (2.7 * np.cos(t), 1.9 * np.sin(t) + 0.12 * np.sin(3 * t)), 26),
    -0.28,
    "cable-net compression ring; node castings by ring-arc distance",
    "architecture",
    stations=31,
)

# -- verification stress ------------------------------------------------------

add(
    "diamond_beyond_critical_cusps",
    [[1.35, 0.0], [0.0, 1.35], [-1.35, 0.0], [0.0, -1.35]],
    1.28,
    "offset just below the corner radius: certified cusps, crowded stations",
    "verification",
    stations=41,
)
add(
    "gear_pocket_cusp_flowers",
    polar(lambda t: 1.9 + 0.24 * np.cos(7 * t), 42),
    0.62,
    "inward offset beyond the lobe radius: cusp pairs kept, never trimmed",
    "verification",
    stations=45,
)


def build(case: DistanceCase) -> CubicPHSplineClosed:
    return CubicPHSplineClosed(case.points)


if __name__ == "__main__":
    run(CASES, build, OUT, "closed cubic PH source")
