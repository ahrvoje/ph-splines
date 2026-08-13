"""Render 32 open cubic-PH offset *distance-query* studies into this folder.

Run from the repository root:
    python examples/cubic_offset_distance/generate_examples.py

Use ``--check-only`` to verify every offset and its distance API without
drawing.  Each case builds a :class:`CubicPHSplineOpen`, compiles one exact
rational NURBS offset with its verified metric certificate, and places all
distance stations with ``point_at_length`` measured along the offset locus
itself - the quantity a machine, vehicle, or beam travelling that locus
actually experiences.
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

from ph_spline import CubicPHSplineOpen  # noqa: E402

OUT = Path(__file__).resolve().parent
CASES: list[DistanceCase] = []


def add(name, points, distance, note, category, **kwargs):
    CASES.append(DistanceCase(name, points, distance, note, category, **kwargs))


# -- semiconductor and electronics ------------------------------------------

add(
    "wafer_saw_street_kerf",
    meander(9.0, lambda x: 0.14 * np.sin(1.15 * x) + 0.02 * x, 16),
    0.09,
    "dicing-blade kerf line: coolant nozzles indexed by true kerf travel",
    "semiconductor",
    stations=31, highlight=(0.30, 0.55),
)
add(
    "reticle_stage_settling_scan",
    open_curve(lambda t: (t, 0.22 * np.sin(2.1 * t) / (1 + 0.3 * t)), 0.0, 6.4, 15),
    -0.12,
    "metrology standoff track with settle checkpoints every equal scan length",
    "semiconductor",
    stations=27,
)
add(
    "probe_card_cleaning_pass",
    seeded_walk(1204, 13, step=0.85, turn_scale=0.33),
    0.16,
    "abrasive-film pass beside a probe row; dwell stations by film travel",
    "semiconductor",
    stations=23, highlight=(0.1, 0.28),
)
add(
    "pcb_rout_tab_channel",
    catmull_rom(
        [(0.0, 0.0), (1.4, 0.55), (2.9, 0.2), (4.2, 0.9), (5.6, 0.4)],
        6, closed=False,
    ),
    0.14,
    "breakaway rout channel: tab positions at exact channel-edge intervals",
    "electronics",
    stations=25, extra=(-0.14,),
)

# -- photonics and precision optics ------------------------------------------

add(
    "photonic_waveguide_s_bend",
    open_curve(lambda t: (t, 0.55 * np.tanh(1.6 * (t - 2.2))), 0.0, 4.4, 17),
    0.07,
    "cladding-edge clearance of an S-bend; loss audit points by edge length",
    "photonics",
    stations=29, highlight=(0.4, 0.6),
)
add(
    "fiber_splice_tray_loop",
    open_curve(lambda t: (2.0 * np.cos(t), 1.35 * np.sin(t)), 0.35, 2.75, 13),
    -0.18,
    "minimum-bend guide wall; strain-relief clips at equal fiber travel",
    "photonics",
    stations=21,
)
add(
    "interferometer_delay_line",
    meander(7.5, lambda x: 0.5 * np.sin(1.35 * x) * np.exp(-0.09 * x), 17),
    0.11,
    "folded optical delay path; picosecond taps by geometric path length",
    "photonics",
    stations=33,
)
add(
    "lithography_illum_homogenizer",
    open_curve(lambda t: (t, 0.32 * np.sin(3.1 * t) * (1 - t / 8.5)), 0.0, 7.0, 19),
    0.10,
    "fly-eye channel wall: facet boundaries at constant wall-arc pitch",
    "photonics",
    stations=36,
)

# -- aerospace ----------------------------------------------------------------

add(
    "wing_leading_edge_deice",
    open_curve(lambda t: (2.6 * t, 0.85 * np.sqrt(np.clip(t, 1e-9, None)) * (1 - 0.35 * t)),
               0.02, 1.0, 13),
    0.08,
    "pneumatic de-ice boot seam offset; stitch stations by true seam length",
    "aerospace",
    stations=25,
)
add(
    "afp_tape_course",
    open_curve(lambda t: (t, 0.6 * np.sin(0.75 * t) + 0.08 * t), 0.0, 8.0, 17),
    0.22,
    "automated fiber placement course centerline and one tow-width offset",
    "aerospace",
    stations=33, extra=(0.44,), highlight=(0.25, 0.375),
)
add(
    "fuselage_frame_shim_line",
    seeded_walk(3305, 15, step=0.8, turn_scale=0.24, drift=0.02),
    -0.15,
    "assembly shim reference line; drill stations at equal reference travel",
    "aerospace",
    stations=29,
)
add(
    "reentry_heatshield_ablator_land",
    open_curve(lambda t: (2.4 * np.sin(t), 1.7 * (1 - np.cos(t))), 0.15, 1.45, 11),
    0.19,
    "ablator tile land line: gap fillers pitched by land-arc distance",
    "aerospace",
    stations=19,
)

# -- manufacturing ------------------------------------------------------------

add(
    "cnc_finish_pass_stepover",
    meander(6.5, lambda x: 0.35 * np.sin(0.95 * x) + 0.05 * x, 15),
    0.18,
    "finish-pass stepover with chip-load probes at constant flute travel",
    "manufacturing",
    stations=27, extra=(0.36, 0.54),
)
add(
    "waterjet_lead_in_taper",
    open_curve(lambda t: (t * np.cos(2.1 * t), t * np.sin(2.1 * t)), 0.35, 2.1, 13),
    0.12,
    "spiral lead-in; taper compensation keyed to exact jet travel",
    "manufacturing",
    stations=23, highlight=(0.55, 0.8),
)
add(
    "laser_cut_bridge_micro_joints",
    catmull_rom(
        [(0.0, 0.0), (1.2, 0.85), (2.7, 0.75), (3.6, -0.2), (5.0, 0.15),
         (6.2, 0.9)],
        6, closed=False,
    ),
    0.10,
    "micro-joint bridges every fixed increment of true cut-edge length",
    "manufacturing",
    stations=37,
)
add(
    "weld_seam_dressing_band",
    seeded_walk(881, 14, step=0.9, turn_scale=0.3),
    0.20,
    "post-weld dressing band; NDT inspection points by seam travel",
    "manufacturing",
    stations=25, extra=(-0.20,), highlight=(0.6, 0.84),
)
add(
    "giga_casting_die_vent_channel",
    meander(5.8, lambda x: 0.28 * np.sin(1.7 * x) * (1 + 0.12 * x), 15),
    -0.13,
    "vacuum vent land beside the die edge; chill vents at equal land travel",
    "manufacturing",
    stations=27,
)

# -- robotics and autonomy ----------------------------------------------------

add(
    "amr_warehouse_dock_approach",
    open_curve(lambda t: (3.0 * t, 1.5 * t * t * (3 - 2 * t)), 0.0, 1.0, 12),
    0.24,
    "docking corridor edge; braking checkpoints by true corridor travel",
    "robotics",
    stations=21, highlight=(0.72, 1.0),
)
add(
    "robot_polish_orbit_standoff",
    open_curve(lambda t: (2.3 * np.cos(t), 1.5 * np.sin(t)), 0.25, 2.9, 15),
    0.30,
    "end-effector standoff path; force-control gains staged by tool travel",
    "robotics",
    stations=25,
)
add(
    "drone_photogrammetry_transect",
    meander(8.4, 0.85, 17, frequency=2.3),
    0.35,
    "side-lap transect offset; shutter stations at constant ground track",
    "robotics",
    stations=41,
)
add(
    "rov_pipeline_inspection_track",
    seeded_walk(4470, 16, step=1.0, turn_scale=0.21, drift=-0.015),
    -0.4,
    "ROV holds a fixed standoff; sonar pings by along-track distance",
    "robotics",
    stations=29, highlight=(0.35, 0.5),
)

# -- medical ------------------------------------------------------------------

add(
    "catheter_vessel_wall_margin",
    seeded_walk(9021, 15, step=0.75, turn_scale=0.34),
    0.14,
    "guidewire wall-clearance margin; radiopaque marks by lumen travel",
    "medical",
    stations=27,
)
add(
    "cochlear_electrode_insertion",
    open_curve(lambda t: ((2.4 - 0.5 * t) * np.cos(t), (2.4 - 0.5 * t) * np.sin(t)),
               0.2, 2.5, 13),
    -0.16,
    "basilar-membrane clearance spiral; contact pitch by insertion depth",
    "medical",
    stations=23,
)
add(
    "radiosurgery_scan_carriage",
    meander(6.0, lambda x: 0.4 * np.sin(1.1 * x) * np.exp(-0.05 * x), 15),
    0.21,
    "collimator carriage rail; dose control points at equal rail travel",
    "medical",
    stations=25, highlight=(0.42, 0.58),
)
add(
    "orthopedic_saw_guide_slot",
    catmull_rom(
        [(0.0, 0.0), (1.1, 0.5), (2.3, 0.65), (3.4, 0.3), (4.4, 0.65)],
        6, closed=False,
    ),
    0.12,
    "patient-specific cutting-slot wall; depth stops by slot-arc distance",
    "medical",
    stations=21, extra=(-0.12,),
)

# -- energy and infrastructure ------------------------------------------------

add(
    "wind_blade_te_protection_strip",
    open_curve(lambda t: (4.2 * t, 0.55 * np.sin(2.6 * t) * (1 - t) + 0.9 * t * (1 - t)),
               0.0, 1.0, 15),
    0.10,
    "trailing-edge erosion strip bond line; adhesive beads by strip travel",
    "energy",
    stations=29,
)
add(
    "hv_catenary_swing_corridor",
    open_curve(lambda t: (t, 0.55 * (np.cosh(t - 2.6) - 1.0)), 0.0, 5.2, 15),
    0.45,
    "conductor blow-out corridor; clearance audits by conductor length",
    "energy",
    stations=27,
)
add(
    "subsea_cable_burial_route",
    seeded_walk(5512, 17, step=1.05, turn_scale=0.19),
    0.55,
    "trencher standoff route; burial-depth logs at fixed route chainage",
    "energy",
    stations=33, highlight=(0.2, 0.32),
)
add(
    "penstock_weld_inspection_line",
    open_curve(lambda t: (2.7 * np.cos(t), 2.0 * np.sin(t)), 0.3, 2.8, 15),
    -0.24,
    "inner-wall inspection ring segment; UT probes by wall-arc distance",
    "energy",
    stations=25,
)

# -- transport ----------------------------------------------------------------

add(
    "maglev_transition_easement",
    open_curve(lambda t: (t, 0.055 * t**3 / (1 + 0.28 * t)), 0.0, 3.6, 13),
    0.16,
    "guideway easement with lateral-jerk gates at equal guideway travel",
    "transport",
    stations=23, highlight=(0.6, 0.8),
)
add(
    "highway_barrier_setback",
    catmull_rom(
        [(0.0, 0.0), (1.6, 0.35), (3.1, 1.15), (4.7, 1.4), (6.3, 2.2)],
        6, closed=False,
    ),
    0.28,
    "crash-barrier setback line; post spacing by true barrier run",
    "transport",
    stations=31,
)
# -- verification stress ------------------------------------------------------

add(
    "cusp_crowding_study",
    open_curve(lambda t: (2.05 * np.cos(t), 1.25 * np.sin(t)), 0.3, 2.85, 15),
    1.05,
    "beyond-critical inward offset: stations crowd toward certified cusps",
    "verification",
    stations=41,
)


def build(case: DistanceCase) -> CubicPHSplineOpen:
    return CubicPHSplineOpen(case.points)


if __name__ == "__main__":
    run(CASES, build, OUT, "open cubic PH source")
