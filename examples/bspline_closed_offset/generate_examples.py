"""Render 32 closed PH B-spline exact-offset studies into this folder.

Run from the repository root:
    python examples/bspline_closed_offset/generate_examples.py

Use ``--check-only`` to construct and verify every offset without drawing.
Each case builds a :class:`PHBSplineClosed` (several at higher continuity
orders), requests exact rational NURBS offsets of degree
``4 * preimage_degree + 1``, and verifies them against
``z(u) + d * N_L(u)`` before publishing an image.
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

from ph_spline import PHBSplineClosed  # noqa: E402

OUT = Path(__file__).resolve().parent
CASES: list[OffsetCase] = []


def add(name, points, distances, note, category, **kwargs):
    CASES.append(OffsetCase(name, points, tuple(distances), note, category,
                            **kwargs))


def smooth_square(values: np.ndarray, sharpness: float = 0.1) -> np.ndarray:
    return values / np.sqrt(values * values + sharpness * sharpness)


# -- turbomachinery and rotating equipment ---------------------------------

add(
    "impeller_passage_walls",
    periodic(
        lambda t: (
            2.0 * np.cos(t) + 0.3 * np.cos(2.0 * t + 0.5),
            1.4 * np.sin(t) + 0.2 * np.sin(2.0 * t),
        ),
        23,
    ),
    (-0.16, 0.16),
    "pressure- and suction-side walls about a pump passage midline",
    "engineering",
)
add(
    "volute_cutwater_clearance",
    polar(lambda t: 1.7 + 0.28 * np.cos(t - 0.9) + 0.07 * np.cos(3.0 * t), 21),
    (-0.14, -0.3, -0.5),
    "cutwater running clearances outside a volute tongue circle",
    "engineering",
)
add(
    "kaplan_blade_shells",
    periodic(
        lambda t: (
            2.15 * np.cos(t) + 0.26 * np.cos(2.0 * t),
            0.95 * np.sin(t) - 0.12 * np.sin(2.0 * t),
        ),
        21,
    ),
    (-0.12, 0.12, 0.26),
    "camber shells about a Kaplan runner blade section",
    "engineering",
    kwargs={"g_order": 3},
)
add(
    "rotary_housing_offsets",
    periodic(
        lambda t: (
            2.15 * np.cos(t) + 0.42 * np.cos(2.0 * t),
            2.15 * np.sin(t) - 0.42 * np.sin(2.0 * t) * 0.35,
        ),
        25,
    ),
    (-0.18, -0.4),
    "apex-seal sweep envelopes outside a rotary-engine bore section",
    "engineering",
)
add(
    "harmonic_wave_generator",
    polar(lambda t: 1.6 + 0.11 * np.cos(2.0 * t), 19),
    (-0.1, 0.1, 0.22),
    "bearing lands about an elliptic wave-generator profile",
    "engineering",
    kwargs={"g_order": 4},
)
add(
    "cam_phaser_lobe",
    polar(lambda t: 1.35 + 0.3 * np.cos(t + 0.5) + 0.09 * np.cos(2.0 * t - 0.4), 21),
    (-0.16, 0.16),
    "vane clearance bands about a cam-phaser lobe midline",
    "engineering",
)
add(
    "turbine_platform_seal",
    polar(lambda t: 1.85 + 0.06 * np.cos(5.0 * t + 1.2), 27),
    (-0.13, -0.28, -0.45),
    "platform seal lands stepped outward from a firtree rim circle",
    "engineering",
)

# -- fusion and accelerators ------------------------------------------------

add(
    "fusion_blanket_shells",
    periodic(
        lambda t: (
            1.1 * np.cos(t + 0.32 * np.sin(t)),
            1.42 * np.sin(t),
        ),
        23,
    ),
    (-0.15, -0.32, -0.52),
    "breeding-blanket shells outside a D-shaped plasma boundary",
    "physics",
)
add(
    "stellarator_coil_band",
    periodic(
        lambda t: (
            1.9 * np.cos(t) + 0.28 * np.cos(3.0 * t),
            1.35 * np.sin(t) + 0.2 * np.sin(3.0 * t + 0.9),
        ),
        27,
    ),
    (-0.16, -0.34),
    "winding-pack band outside a shaped stellarator coil filament",
    "physics",
    kwargs={"g_order": 3},
)
add(
    "rf_cavity_aperture",
    polar(lambda t: 1.5 + 0.16 * np.cos(4.0 * t), 25),
    (0.16, 0.34),
    "iris aperture reductions inside an RF cavity wall section",
    "physics",
)
add(
    "detector_barrel_layers",
    polar(lambda t: 1.45 + 0.045 * np.cos(6.0 * t + 0.7), 29),
    (-0.2, -0.42, -0.68, -0.98),
    "silicon tracker barrel layers grown from a beam-pipe waist",
    "physics",
)
add(
    "quadrupole_pole_gap",
    periodic(
        lambda t: (
            1.8 * np.cos(t) * (1.0 + 0.12 * np.cos(4.0 * t)),
            1.8 * np.sin(t) * (1.0 - 0.12 * np.cos(4.0 * t)),
        ),
        29,
    ),
    (-0.14, 0.14),
    "pole-face gap bands about a quadrupole aperture contour",
    "physics",
)

# -- space engineering ------------------------------------------------------

add(
    "solar_array_keepout",
    periodic(
        lambda t: (
            2.3 * smooth_square(np.cos(t), 0.5),
            1.5 * smooth_square(np.sin(t), 0.5),
        ),
        25,
    ),
    (-0.24, -0.52, -0.85),
    "articulation keepout rings around a solar array wing outline",
    "space",
)
add(
    "crater_rim_terraces",
    polar(
        lambda t: 1.75
        + 0.22 * np.cos(2.0 * t + 0.6)
        + 0.1 * np.cos(5.0 * t)
        + 0.05 * np.cos(9.0 * t + 1.4),
        31,
    ),
    (0.24, 0.5, 0.8),
    "descent terraces cut inside an irregular crater rim",
    "space",
)
add(
    "heat_shield_tile_lands",
    periodic(
        lambda t: (
            2.2 * np.cos(t) - 0.3 * np.cos(2.0 * t),
            1.6 * np.sin(t) + 0.14 * np.sin(2.0 * t),
        ),
        23,
    ),
    (-0.13, 0.13),
    "tile-gap filler lands about a capsule shield perimeter",
    "space",
)
add(
    "lunar_pad_blast_rings",
    polar(lambda t: 1.3 + 0.14 * np.cos(3.0 * t - 0.4), 21),
    (-0.35, -0.75, -1.2, -1.7),
    "ejecta blast-protection rings around a landing pad apron",
    "space",
)

# -- manufacturing ----------------------------------------------------------

add(
    "mems_comb_gap",
    polar(
        lambda t: 1.6 * (1.0 + 0.05 * smooth_square(np.sin(9.0 * t), 0.16)),
        41,
    ),
    (-0.09, 0.09),
    "etch-gap walls about a MEMS comb-drive rotor outline",
    "manufacturing",
)
add(
    "stamping_binder_wrap",
    periodic(
        lambda t: (
            2.4 * np.cos(t) + 0.26 * np.cos(2.0 * t - 0.7),
            1.5 * np.sin(t) + 0.2 * np.sin(3.0 * t),
        ),
        23,
    ),
    (-0.2, -0.44),
    "binder wrap addendum bands outside a drawn panel opening",
    "manufacturing",
)
add(
    "additive_contour_shells",
    catmull_rom(
        [(-2.3, -1.0), (-0.8, -1.5), (1.2, -1.25), (2.6, -1.35), (2.95, 0.1),
         (1.8, 1.25), (-0.2, 1.55), (-2.1, 1.2), (-2.85, 0.15)],
        3,
    ),
    (0.2, 0.42, 0.66),
    "inner contour shells of an additive-manufactured wall section",
    "manufacturing",
    fill=True,
)
add(
    "injection_cooling_circuit",
    periodic(
        lambda t: (
            2.1 * np.cos(t) - 0.34 * np.cos(2.0 * t),
            1.45 * np.sin(t) + 0.12 * np.sin(2.0 * t),
        ),
        21,
    ),
    (-0.3, -0.62),
    "conformal cooling channel loops outside a cavity insert profile",
    "manufacturing",
)
add(
    "vibratory_bowl_track",
    polar(lambda t: 1.5 + 0.09 * np.cos(1.0 * t + 0.3) + 0.05 * np.cos(4.0 * t), 25),
    (-0.2, -0.42, -0.66),
    "spiral feeder track walls stepped from a bowl hub circle",
    "manufacturing",
)

# -- mobility and logistics -------------------------------------------------

add(
    "karting_track_kerbs",
    catmull_rom(
        [(-2.7, -0.4), (-1.7, -1.3), (0.0, -1.5), (1.5, -0.95), (2.75, -0.5),
         (2.5, 0.7), (1.1, 0.85), (0.1, 1.45), (-1.4, 1.2), (-2.6, 0.6)],
        3,
    ),
    (-0.24, 0.24),
    "kerb lines each side of a karting circuit centerline",
    "mobility",
)
add(
    "city_ring_roads",
    polar(
        lambda t: 1.9
        + 0.3 * np.cos(2.0 * t + 1.1)
        + 0.12 * np.cos(3.0 * t - 0.4),
        25,
    ),
    (-0.3, -0.65, -1.05),
    "orbital ring roads traced outward from an old-town boundary",
    "logistics",
)
add(
    "warehouse_agv_loops",
    periodic(
        lambda t: (
            2.6 * smooth_square(np.cos(t), 0.58),
            1.4 * smooth_square(np.sin(t), 0.58),
        ),
        27,
    ),
    (-0.28, 0.28, 0.58),
    "inner express and outer service AGV loops about a spine circuit",
    "logistics",
)
add(
    "port_turning_basin",
    polar(lambda t: 1.7 + 0.2 * np.cos(t + 0.8) + 0.08 * np.cos(3.0 * t), 23),
    (-0.35, -0.75),
    "tug clearance margins around a turning-basin boundary",
    "logistics",
)

# -- environment and design -------------------------------------------------

add(
    "pond_liner_trench",
    periodic(
        lambda t: (
            2.15 * np.cos(t) + 0.24 * np.cos(2.0 * t - 0.5),
            1.5 * np.sin(t) + 0.18 * np.sin(3.0 * t + 0.7),
        ),
        23,
    ),
    (-0.22, -0.48),
    "liner anchor trench rings outside an ornamental pond shore",
    "geoscience",
)
add(
    "vineyard_frost_fans",
    polar(
        lambda t: 1.8
        + 0.24 * np.cos(2.0 * t)
        + 0.09 * np.cos(5.0 * t + 0.8),
        27,
    ),
    (0.3, 0.62),
    "frost-fan coverage contours inside a hillside vineyard block",
    "geoscience",
)
add(
    "guitar_rosette_rings",
    polar(lambda t: 1.25 + 0.035 * np.cos(5.0 * t), 21),
    (-0.14, -0.3, -0.48, -0.68),
    "inlay rosette rings grown from a soundhole rim",
    "design",
)
add(
    "art_deco_medallion",
    polar(lambda t: 1.7 + 0.17 * np.cos(8.0 * t), 49),
    (-0.2, -0.44),
    "stepped bezel bands outside an eight-lobe deco medallion",
    "design",
)

# -- abstract ---------------------------------------------------------------

add(
    "reuleaux_rotor_bands",
    polar(lambda t: 1.65 + 0.19 * np.cos(3.0 * t), 25),
    (-0.2, 0.2, 0.44),
    "clearance bands about a rounded Reuleaux-like rotor",
    "abstract",
    kwargs={"g_order": 3},
)
add(
    "supershape_cascade",
    polar(lambda t: 1.85 + 0.26 * np.cos(6.0 * t), 49),
    (0.34, 0.7, 1.1),
    "inward cascade past the curvature radius: cusps and loops kept",
    "abstract",
)
add(
    "double_loop_offsets",
    [
        [
            float((2.0 + np.cos(0.5 * t)) * np.cos(t)),
            float((2.0 + np.cos(0.5 * t)) * np.sin(t)),
        ]
        for t in np.linspace(0.0, 4.0 * np.pi, 18, endpoint=False)
    ],
    (0.16, -0.16),
    "turning-number-two closed curve with a periodic preimage lift",
    "abstract",
)


def build(case: OffsetCase) -> PHBSplineClosed:
    return PHBSplineClosed(case.points, **case.kwargs)


if __name__ == "__main__":
    run(CASES, build, OUT, "closed PH B-spline source")
