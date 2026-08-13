"""Render 32 closed PH B-spline offset distance-query studies here.

Run from the repository root:
    python examples/bspline_closed_offset_distance/generate_examples.py

Use ``--check-only`` to verify every offset and its distance API without
drawing.  Each case builds a :class:`PHBSplineClosed` loop (several at
raised continuity order), compiles the exact offset NURBS with its verified
metric certificate, and lays out distance stations with ``point_at_length``
over the one-traversal distance domain of the offset loop.
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

from ph_spline import PHBSplineClosed  # noqa: E402

OUT = Path(__file__).resolve().parent
CASES: list[DistanceCase] = []


def add(name, points, distance, note, category, **kwargs):
    CASES.append(DistanceCase(name, points, distance, note, category, **kwargs))


# -- machining and tooling ---------------------------------------------------

add(
    "pocket_finish_boundary",
    catmull_rom(
        [(2.2, 0.0), (1.5, 1.4), (-0.3, 1.9), (-2.0, 1.0), (-2.3, -0.6),
         (-0.9, -1.8), (1.1, -1.6)],
        5,
    ),
    -0.24,
    "climb-finish boundary inside a pocket; feed overrides by wall travel",
    "machining",
    stations=31, fill=True,
)
add(
    "mold_parting_line_vent_ring",
    polar(lambda t: 2.0 + 0.16 * np.cos(5 * t), 30),
    0.18,
    "vent land outside a parting line; vent slots at equal land travel",
    "machining",
    stations=33,
)
add(
    "trim_die_steel_line",
    periodic(lambda t: (2.3 * np.cos(t) + 0.28 * np.cos(3 * t),
                        1.7 * np.sin(t)), 24),
    0.15,
    "trim-steel wall line; section grinds by cutting-edge travel",
    "machining",
    stations=29,
)
add(
    "edm_wire_skim_loop",
    polar(lambda t: 1.75 + 0.09 * np.cos(7 * t), 35),
    -0.1,
    "wire-EDM skim loop; spark-gap logs at constant wire travel",
    "machining",
    stations=37, highlight=(0.2, 0.32),
)

# -- sealing and fluid handling ----------------------------------------------

add(
    "fip_gasket_dispense_loop",
    catmull_rom(
        [(1.9, 0.2), (1.0, 1.5), (-0.7, 1.7), (-1.9, 0.7), (-1.7, -0.9),
         (-0.1, -1.7), (1.4, -1.3)],
        5,
    ),
    0.13,
    "form-in-place gasket bead; dispense rate scheduled by bead travel",
    "sealing",
    stations=29,
)
add(
    "hydraulic_manifold_o_ring",
    periodic(lambda t: (1.6 * np.cos(t), 1.15 * np.sin(t) + 0.1 * np.sin(2 * t)), 18),
    0.12,
    "gland wall around a manifold port; volume audit by gland travel",
    "sealing",
    stations=23,
)
add(
    "fuel_cell_bipolar_seal",
    polar(lambda t: 2.1 + 0.2 * np.cos(2 * t) + 0.06 * np.cos(6 * t), 26),
    -0.14,
    "membrane seal line inside the flow-field border",
    "sealing",
    stations=31,
)
add(
    "cryo_transfer_line_bellows",
    polar(lambda t: 1.7 + 0.05 * np.cos(9 * t), 36),
    0.16,
    "bellows convolution root ring; fatigue gauges by ring travel",
    "cryogenics",
    stations=37,
)

# -- electric machines and magnetics -----------------------------------------

add(
    "axial_flux_stator_ring",
    polar(lambda t: 2.05 + 0.13 * np.cos(10 * t), 40),
    0.14,
    "axial-flux stator boundary; coil terminations by rim travel",
    "e-machines",
    stations=41, fill=True,
)
add(
    "mri_gradient_coil_track",
    periodic(lambda t: (2.5 * np.cos(t), 1.6 * np.sin(t) + 0.14 * np.sin(3 * t)), 26),
    -0.2,
    "gradient conductor track; joint braze points by conductor length",
    "magnetics",
    stations=29,
)
add(
    "transformer_core_window",
    periodic(lambda t: (1.9 * np.cos(t) + 0.42 * np.cos(3 * t),
                        1.5 * np.sin(t) + 0.2 * np.sin(3 * t)), 24),
    0.17,
    "winding clearance inside a core window",
    "magnetics",
    stations=27,
)
add(
    "halbach_array_keeper_ring",
    polar(lambda t: 1.8 + 0.08 * np.cos(12 * t), 48),
    0.11,
    "keeper ring outside a Halbach array; segment gaps by ring travel",
    "magnetics",
    stations=49, kwargs={"g_order": 4},
)

# -- aerospace and marine ------------------------------------------------------

add(
    "turbofan_acoustic_liner_ring",
    polar(lambda t: 2.4 + 0.05 * np.cos(3 * t), 24),
    -0.26,
    "inlet acoustic liner ring; splice locations by duct-wall travel",
    "aerospace",
    stations=29,
)
add(
    "propeller_hub_fairing_section",
    periodic(lambda t: (1.9 * np.cos(t), 1.25 * np.sin(t) + 0.16 * np.sin(2 * t)), 20),
    0.2,
    "fairing section shell; rivet rows at equal shell-arc distance",
    "marine",
    stations=27,
)
add(
    "airship_frame_ring",
    polar(lambda t: 2.6 + 0.07 * np.cos(4 * t), 28),
    -0.3,
    "envelope frame inner ring; cable lugs by ring circumference",
    "aerospace",
    stations=33,
)
add(
    "auv_pressure_hull_frame",
    periodic(lambda t: (2.1 * np.cos(t), 1.55 * np.sin(t)), 18),
    -0.22,
    "ring-stiffener toe line inside a pressure hull section",
    "marine",
    stations=25, highlight=(0.4, 0.55),
)

# -- medical and biotech -------------------------------------------------------

add(
    "stent_ring_strut_loop",
    polar(lambda t: 1.5 + 0.16 * np.cos(8 * t), 32),
    0.09,
    "laser-cut strut ring; crown positions at constant strut travel",
    "medical",
    stations=33,
)
add(
    "dental_arch_aligner_margin",
    periodic(lambda t: (1.9 * np.cos(t), 1.35 * np.sin(t) + 0.22 * np.sin(2 * t) ** 2), 22),
    0.12,
    "trim margin around a dental arch; scallop points by margin travel",
    "medical",
    stations=29,
)
add(
    "bioreactor_impeller_shroud",
    polar(lambda t: 1.85 + 0.12 * np.cos(6 * t), 30),
    -0.15,
    "shear-protective shroud line; sampling ports by shroud travel",
    "biotech",
    stations=31,
)
add(
    "microfluidic_racetrack_channel",
    periodic(lambda t: (2.4 * np.cos(t), 1.0 * np.sin(t)), 20),
    -0.18,
    "racetrack channel wall; droplet timing gates by channel travel",
    "biotech",
    stations=27, kwargs={"g_order": 4},
)

# -- energy ---------------------------------------------------------------------

add(
    "flywheel_burst_containment",
    polar(lambda t: 2.2 + 0.04 * np.cos(2 * t), 20),
    0.28,
    "burst-liner standoff ring; strain rosettes by liner travel",
    "energy",
    stations=27,
)
add(
    "molten_salt_pipe_trace_loop",
    catmull_rom(
        [(2.3, 0.1), (1.3, 1.6), (-0.5, 2.0), (-2.1, 1.1), (-2.4, -0.7),
         (-0.8, -1.9), (1.2, -1.7)],
        5,
    ),
    0.2,
    "heat-trace lay line around a salt pipe; clamps by trace travel",
    "energy",
    stations=31,
)
add(
    "pv_tracker_torque_tube_ring",
    polar(lambda t: 1.7 + 0.06 * np.cos(5 * t), 25),
    0.13,
    "bearing land ring on a torque tube",
    "energy",
    stations=27,
)
add(
    "smr_containment_penetration",
    periodic(lambda t: (1.75 * np.cos(t), 1.45 * np.sin(t)), 16),
    0.24,
    "penetration reinforcement boundary; studs at equal boundary travel",
    "energy",
    stations=25,
)

# -- consumer, sport, architecture ---------------------------------------------

add(
    "velodrome_measurement_line",
    periodic(lambda t: (2.9 * np.cos(t), 1.75 * np.sin(t)), 24),
    0.25,
    "UCI measurement line offset from the track datum; timing loops",
    "sport",
    stations=33, fill=True, highlight=(0.0, 0.25),
)
add(
    "headphone_ear_cushion_seam",
    periodic(lambda t: (1.6 * np.cos(t), 1.2 * np.sin(t) + 0.14 * np.sin(2 * t)), 18),
    0.1,
    "cushion seam allowance; stitch count from exact seam length",
    "consumer",
    stations=25,
)
add(
    "watch_case_gasket_groove",
    polar(lambda t: 1.45 + 0.05 * np.cos(6 * t), 24),
    -0.09,
    "case-back groove wall; torque checks by groove travel",
    "consumer",
    stations=25,
)
add(
    "arena_bowl_edge_beam",
    periodic(lambda t: (2.8 * np.cos(t) + 0.2 * np.cos(2 * t),
                        2.0 * np.sin(t)), 26),
    -0.35,
    "bowl edge-beam centerline; precast segment joints by beam chainage",
    "architecture",
    stations=31,
)

# -- science and verification ----------------------------------------------------

add(
    "storage_ring_vacuum_wall",
    periodic(lambda t: (2.7 * np.cos(t), 1.2 * np.sin(t)), 22),
    -0.16,
    "storage-ring wall; photon-absorber lips by wall travel",
    "accelerators",
    stations=31,
)
add(
    "ion_trap_electrode_loop",
    polar(lambda t: 1.6 + 0.1 * np.cos(4 * t), 24),
    0.12,
    "RF electrode edge loop; compensation probes by edge travel",
    "science",
    stations=27, kwargs={"g_order": 4},
)
add(
    "high_order_seal_ring_g6",
    polar(lambda t: 1.9 + 0.07 * np.cos(3 * t), 21),
    0.15,
    "G6 seal ring (degree-25 rational offset); leak checks by ring travel",
    "verification",
    stations=25, kwargs={"g_order": 6},
)
add(
    "lobed_cam_beyond_critical",
    polar(lambda t: 1.85 + 0.3 * np.cos(5 * t), 40),
    0.55,
    "inward offset beyond the lobe radius: certified cusp pairs kept",
    "verification",
    stations=45,
)


def build(case: DistanceCase) -> PHBSplineClosed:
    return PHBSplineClosed(case.points, **case.kwargs)


if __name__ == "__main__":
    run(CASES, build, OUT, "closed PH B-spline source")
