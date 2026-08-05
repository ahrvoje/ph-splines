"""Render 128 deliberately varied closed cubic-PH examples into this folder.

Run from the repository root:
    python examples/cubic_closed/generate_examples.py

Use ``--check-only`` to compile and verify the complete corpus without drawing.
Every case is passed directly to :class:`CubicPHSplineClosed`.  The generator
independently checks node interpolation, all declared cyclic joins, the G2
seam, and distance inversion before publishing an image.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ph_spline import CubicPHSplineClosed, InterpolationDomainError

OUT = Path(__file__).resolve().parent
PAGE = "#f9f9f7"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK = "#172033"
MUTED = "#8a94a6"
POINT = "#eb6834"
STATION = "#16856b"
CATEGORY_COLORS = {
    "engineering": "#246fc2",
    "physics": "#7557b7",
    "mathematics": "#167b6b",
    "illustration": "#c45b31",
    "adapted": "#2a78d6",
}


@dataclass(frozen=True)
class Case:
    name: str
    points: list[list[float]]
    note: str
    category: str
    style: str = "outline"


def periodic(
    function: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    count: int,
    *,
    phase: float = 0.017,
) -> list[list[float]]:
    t = phase + np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    x, y = function(t)
    return np.column_stack((x, y)).astype(np.float64).tolist()


def polar(
    radius: Callable[[np.ndarray], np.ndarray],
    count: int,
    *,
    phase: float = 0.017,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
) -> list[list[float]]:
    return periodic(
        lambda t: (
            x_scale * radius(t) * np.cos(t),
            y_scale * radius(t) * np.sin(t),
        ),
        count,
        phase=phase,
    )


def catmull_rom(keys: list[tuple[float, float]], samples: int = 5) -> list[list[float]]:
    """Sample a periodic Catmull-Rom design polygon without duplicating its seam."""
    values = np.asarray(keys, dtype=np.float64)
    result: list[list[float]] = []
    count = len(values)
    for index in range(count):
        p0 = values[(index - 1) % count]
        p1 = values[index]
        p2 = values[(index + 1) % count]
        p3 = values[(index + 2) % count]
        for sample in range(samples):
            u = sample / samples
            point = 0.5 * (
                2.0 * p1
                + (-p0 + p2) * u
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u**2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u**3
            )
            result.append(point.tolist())
    return result


def superellipse(power: float, x_scale: float, y_scale: float, count: int) -> list[list[float]]:
    def evaluate(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c, s = np.cos(t), np.sin(t)
        denominator = (np.abs(c) ** power + np.abs(s) ** power) ** (1.0 / power)
        return x_scale * c / denominator, y_scale * s / denominator

    return periodic(evaluate, count)


def smooth_square(values: np.ndarray, sharpness: float = 0.08) -> np.ndarray:
    return values / np.sqrt(values * values + sharpness * sharpness)


def cassini(count: int = 113) -> list[list[float]]:
    a, b = 0.62, 1.0

    def radius(t: np.ndarray) -> np.ndarray:
        c2 = np.cos(2.0 * t)
        return np.sqrt(a * a * c2 + np.sqrt(b**4 - a**4 * np.sin(2.0 * t) ** 2))

    return polar(radius, count)


def fourier_blob(
    coefficients: tuple[tuple[float, int, float], ...],
    count: int,
    *,
    base: float = 1.0,
) -> list[list[float]]:
    return polar(
        lambda t: base
        + sum(amplitude * np.cos(frequency * t + phase) for amplitude, frequency, phase in coefficients),
        count,
    )


CASES: list[Case] = []


def add(
    name: str,
    points: list[list[float]],
    note: str,
    category: str,
    style: str = "outline",
) -> None:
    CASES.append(Case(name, points, note, category, style))


# 1-32: real design and engineering geometries.
add(
    "grand_prix_circuit",
    catmull_rom(
        [
            (-2.8, -0.3), (-1.9, -1.2), (-0.4, -1.35), (0.7, -0.75),
            (1.7, -1.15), (2.8, -0.35), (2.45, 0.75), (1.45, 0.62),
            (0.8, 1.35), (-0.25, 1.1), (-0.9, 0.45), (-1.85, 0.85),
            (-2.75, 0.45),
        ],
        5,
    ),
    "motorsport centerline with sweepers, a chicane and two hairpins",
    "engineering",
    "track",
)
add(
    "turbine_blade_section",
    periodic(
        lambda t: (
            1.42 * np.cos(t) + 0.24 * np.cos(2 * t) - 0.07 * np.cos(3 * t),
            0.34 * np.sin(t) + 0.10 * np.sin(2 * t) + 0.025 * np.sin(3 * t),
        ),
        109,
    ),
    "closed axial-turbine blade section with camber and finite trailing radius",
    "engineering",
    "profile",
)
add(
    "tokamak_flux_surface",
    periodic(
        lambda t: (
            1.05 * np.cos(t + 0.30 * np.sin(t)),
            1.34 * np.sin(t),
        ),
        103,
    ),
    "D-shaped magnetic-confinement flux surface with triangularity",
    "physics",
    "field",
)
add(
    "precision_cam_profile",
    polar(lambda t: 1.0 + 0.27 * np.cos(t) + 0.075 * np.cos(2 * t - 0.35), 97),
    "asymmetric rise, dwell and return profile for a precision cam",
    "engineering",
    "profile",
)
add(
    "noncircular_gear",
    polar(
        lambda t: (1.0 + 0.17 * np.cos(2 * t))
        * (1.0 + 0.052 * smooth_square(np.sin(14 * t), 0.12)),
        197,
    ),
    "elliptic pitch modulation with fourteen smoothly resolved teeth",
    "engineering",
    "profile",
)
add(
    "three_body_choreography",
    periodic(lambda t: (1.25 * np.sin(t), 0.72 * np.sin(t) * np.cos(t)), 181),
    "figure-eight periodic orbit associated with equal-mass three-body choreography",
    "physics",
    "winding",
)
add(
    "hydrofoil_section",
    periodic(
        lambda t: (
            1.55 * np.cos(t) + 0.13 * np.cos(2 * t),
            0.27 * np.sin(t) + 0.055 * np.sin(2 * t) - 0.025 * np.sin(3 * t),
        ),
        101,
    ),
    "low-drag hydrofoil section with reflexed aft camber",
    "engineering",
    "profile",
)
add(
    "wankel_rotor_housing",
    periodic(
        lambda t: (np.cos(t) + 0.22 * np.cos(2 * t), np.sin(t) - 0.22 * np.sin(2 * t)),
        107,
    ),
    "three-lobed epitrochoid-inspired rotary-engine housing",
    "engineering",
    "profile",
)
add(
    "cycloidal_drive_disc",
    fourier_blob(((0.095, 11, 0.0), (0.026, 22, 0.4)), 179),
    "eleven-lobe cycloidal reducer disc with second-harmonic relief",
    "engineering",
    "profile",
)
add(
    "wave_washer",
    polar(lambda t: 1.0 + 0.095 * np.sin(6 * t), 127, x_scale=1.16, y_scale=0.82),
    "six-wave spring-washer planform",
    "engineering",
    "profile",
)
add(
    "bearing_race_waviness",
    fourier_blob(((0.012, 17, 0.1), (0.006, 29, 1.0), (0.003, 41, -0.4)), 211),
    "exaggerated multi-order roundness error on a bearing race",
    "engineering",
    "metrology",
)
add(
    "harmonic_drive_flexspline",
    polar(lambda t: (1.0 + 0.085 * np.cos(2 * t)) * (1.0 + 0.025 * np.cos(32 * t)), 257),
    "elliptically strained flexspline carrying thirty-two teeth",
    "engineering",
    "profile",
)
add(
    "nonround_gasket",
    superellipse(5.0, 1.45, 0.86, 109),
    "rounded-rectangular sealing contour for a manifold flange",
    "engineering",
    "profile",
)
add(
    "pressure_vessel_capsule",
    catmull_rom([(-1.7, 0.0), (-1.45, -0.65), (0.0, -0.78), (1.45, -0.65),
                 (1.7, 0.0), (1.45, 0.65), (0.0, 0.78), (-1.45, 0.65)], 7),
    "smooth capsule section with continuously rounded pressure heads",
    "engineering",
    "profile",
)
add(
    "heat_exchanger_header",
    periodic(lambda t: (1.65 * np.cos(t), 0.56 * np.sin(t) + 0.07 * np.sin(7 * t)), 137),
    "elongated heat-exchanger header with distributed tube-seat corrugation",
    "engineering",
    "profile",
)
add(
    "impeller_shroud",
    fourier_blob(((0.13, 7, 0.45), (0.038, 14, -0.15)), 151),
    "seven-passage centrifugal-impeller shroud envelope",
    "engineering",
    "profile",
)
add(
    "pump_volute_casing",
    catmull_rom([(-0.2, -0.25), (0.65, -0.55), (1.45, -0.15), (1.72, 0.65),
                 (1.05, 1.3), (0.05, 1.35), (-0.85, 0.9), (-1.3, 0.15),
                 (-1.0, -0.7)], 6),
    "single-scroll centrifugal-pump volute outline",
    "engineering",
    "profile",
)
add(
    "combustion_chamber",
    catmull_rom([(-0.8, -1.2), (-1.05, -0.35), (-0.95, 0.75), (-0.45, 1.3),
                 (0.4, 1.1), (0.95, 0.45), (0.85, -0.45), (0.35, -1.25)], 7),
    "asymmetric high-tumble combustion-chamber section",
    "engineering",
    "profile",
)
add(
    "rocket_nozzle_section",
    catmull_rom([(-1.8, -0.22), (-0.55, -0.35), (0.0, -0.72), (0.9, -1.05),
                 (1.8, -1.18), (1.8, 1.18), (0.9, 1.05), (0.0, 0.72),
                 (-0.55, 0.35), (-1.8, 0.22)], 5),
    "closed wall section through a bell nozzle and combustion throat",
    "engineering",
    "profile",
)
add(
    "robot_workcell_boundary",
    catmull_rom([(-2.0, -1.0), (-0.8, -1.35), (0.2, -0.9), (1.25, -1.25),
                 (2.0, -0.4), (1.75, 0.7), (0.75, 1.25), (-0.25, 0.8),
                 (-1.35, 1.2), (-2.15, 0.35)], 5),
    "irregular robot safety envelope around fixtures and operator access",
    "engineering",
    "track",
)
add(
    "autonomous_test_track",
    catmull_rom([(-2.6, 0.1), (-2.1, -1.0), (-0.8, -1.25), (0.1, -0.45),
                 (1.0, -1.0), (2.45, -0.65), (2.7, 0.45), (1.6, 1.2),
                 (0.35, 0.75), (-0.55, 1.35), (-1.7, 0.95)], 5),
    "closed proving-ground route for repeated autonomous-vehicle trials",
    "engineering",
    "track",
)
add(
    "oval_chicane_track",
    catmull_rom([(-2.5, 0.0), (-1.8, -0.9), (-0.2, -0.9), (0.6, -0.35),
                 (1.1, -0.95), (2.35, -0.65), (2.65, 0.2), (1.8, 0.95),
                 (0.25, 0.85), (-0.65, 0.3), (-1.2, 0.95), (-2.3, 0.65)], 5),
    "speedway oval interrupted by a closed infield chicane",
    "engineering",
    "track",
)
add(
    "compliant_gripper_frame",
    catmull_rom([(-1.5, -1.0), (-0.45, -0.75), (-0.25, 0.15), (-0.85, 0.95),
                 (0.0, 0.55), (0.85, 0.95), (0.25, 0.15), (0.45, -0.75),
                 (1.5, -1.0), (0.0, -1.35)], 6),
    "single-piece flexure gripper frame with two compliant jaws",
    "engineering",
    "profile",
)
add(
    "flexure_link_outline",
    periodic(lambda t: ((1.2 + 0.34 * np.cos(2 * t)) * np.cos(t),
                        (0.72 - 0.22 * np.cos(2 * t)) * np.sin(t)), 109),
    "dog-bone flexure link with narrowed elastic waist",
    "engineering",
    "profile",
)
add(
    "pcb_serpentine_return",
    periodic(lambda t: (1.75 * np.cos(t), 0.72 * np.sin(t) + 0.17 * np.sin(5 * t)), 151),
    "length-matched serpentine interconnect closed through its return lane",
    "engineering",
    "track",
)
add(
    "microfluidic_mixer_loop",
    periodic(lambda t: (1.4 * np.sin(2 * t + 0.9),
                        0.82 * np.sin(3 * t) + 0.10 * np.sin(t)), 211),
    "multi-winding passive-mixer channel crossing between alternating lobes",
    "engineering",
    "winding",
)
add(
    "peristaltic_pump_track",
    fourier_blob(((0.18, 3, 0.2), (0.07, 5, -0.6), (0.035, 9, 0.8)), 137),
    "closed elastomeric tube path around a three-roller pump",
    "engineering",
    "track",
)
add(
    "robot_cable_service_loop",
    periodic(lambda t: (1.2 * np.cos(t) + 0.38 * np.cos(4 * t + 0.3),
                        0.9 * np.sin(t) - 0.28 * np.sin(3 * t)), 157),
    "closed cable-service loop with controlled crossover clearance",
    "engineering",
    "winding",
)
add(
    "turbine_disk_firtree_rim",
    fourier_blob(((0.055, 12, 0.0), (0.022, 24, math.pi), (0.009, 36, 0.3)), 229),
    "periodic fir-tree attachment relief around a turbine disk",
    "engineering",
    "profile",
)
add(
    "fan_duct_liner",
    superellipse(3.2, 1.35, 1.0, 127),
    "rounded-square fan duct suitable for compact equipment packaging",
    "engineering",
    "profile",
)
add(
    "marine_propeller_envelope",
    fourier_blob(((0.24, 4, -0.45), (0.065, 8, 0.8), (0.035, 3, 0.1)), 173),
    "skewed four-blade marine-propeller planform envelope",
    "engineering",
    "profile",
)
add(
    "drone_propeller_envelope",
    fourier_blob(((0.31, 3, 0.35), (0.08, 6, -0.2)), 149),
    "three-blade compact-drone propeller clearance envelope",
    "engineering",
    "profile",
)

# 33-56: physics, dynamics and multi-winding periodic trajectories.
add("plasma_m2n1_mode", fourier_blob(((0.16, 2, 0.2), (0.055, 5, -0.5)), 139),
    "perturbed plasma boundary carrying coupled low-order modes", "physics", "field")
add("magnetic_island_separatrix", periodic(lambda t: (1.35 * np.sin(t), 0.68 * np.sin(2 * t)), 173),
    "two-lobed magnetic-island separatrix projection", "physics", "winding")
add("elliptic_polarization", periodic(lambda t: (1.35 * np.cos(t), 0.55 * np.sin(t + 0.22)), 73),
    "phase-shifted elliptic polarization state", "physics", "orbit")
add("duffing_periodic_orbit", periodic(lambda t: (1.15 * np.cos(t) + 0.28 * np.cos(3 * t),
                                                    0.82 * np.sin(t) - 0.24 * np.sin(3 * t)), 157),
    "nonlinear Duffing-type periodic phase orbit", "physics", "orbit")
add("van_der_pol_limit_cycle", periodic(lambda t: (1.6 * np.cos(t) - 0.22 * np.cos(3 * t),
                                                   1.05 * np.sin(t) - 0.11 * np.sin(3 * t)), 151),
    "Fourier approximation of a relaxation-oscillator limit cycle", "physics", "orbit")
add("pendulum_phase_orbit", periodic(lambda t: (1.55 * np.sin(t), 0.95 * np.cos(t) + 0.18 * np.cos(3 * t)), 131),
    "closed libration orbit in pendulum phase space", "physics", "orbit")
add("cyclotron_epicycle", periodic(lambda t: (np.cos(t) + 0.075 * np.cos(7 * t),
                                              np.sin(t) + 0.075 * np.sin(7 * t)), 181),
    "charged-particle gyromotion superposed on a circular drift", "physics", "winding")
add("precessing_orbit_rosette", polar(lambda t: 1.0 + 0.31 * np.cos(5 * t + 0.2), 179),
    "five-apsis closed surrogate for an orbit under precession", "physics", "orbit")
add("gravitational_slingshot", periodic(lambda t: (1.25 * np.sin(t) + 0.22 * np.sin(2 * t),
                                                    0.62 * np.sin(2 * t) - 0.16 * np.sin(3 * t)), 193),
    "self-crossing return trajectory through a two-body encounter", "physics", "winding")
add("vortex_filament_projection", periodic(lambda t: ((1.0 + 0.50 * np.cos(7 * t)) * np.cos(2 * t),
                                                      (1.0 + 0.50 * np.cos(7 * t)) * np.sin(2 * t)), 263),
    "multi-winding projection of a periodic vortex filament", "physics", "winding")
add("torus_knot_2_3_projection", periodic(lambda t: ((1.0 + 0.48 * np.cos(3 * t)) * np.cos(2 * t),
                                                     (1.0 + 0.48 * np.cos(3 * t)) * np.sin(2 * t)), 233),
    "planar projection of a (2,3) torus knot", "physics", "winding")
add("torus_knot_3_5_projection", periodic(lambda t: ((1.0 + 0.42 * np.cos(5 * t)) * np.cos(3 * t),
                                                     (1.0 + 0.42 * np.cos(5 * t)) * np.sin(3 * t)), 257),
    "planar projection of a (3,5) torus knot", "physics", "winding")
add("figure_eight_knot_projection", periodic(lambda t: ((1.0 + 0.52 * np.cos(4 * t)) * np.cos(3 * t),
                                                        (1.0 + 0.52 * np.cos(4 * t)) * np.sin(3 * t)), 251),
    "multi-winding figure-eight-knot style projection", "physics", "winding")
add("optical_nephroid_caustic", periodic(lambda t: (3 * np.cos(t) - 0.82 * np.cos(3 * t),
                                                     3 * np.sin(t) - 0.82 * np.sin(3 * t)), 173),
    "regularized nephroid envelope of reflected rays", "physics", "field")
add("acoustic_mode_2_5", periodic(lambda t: (1.1 * np.sin(2 * t + 0.31), 0.85 * np.sin(5 * t)), 227),
    "closed 2:5 acoustic-mode interference trace", "physics", "winding")
add("resonance_mode_4_7", periodic(lambda t: ((1.0 + 0.46 * np.cos(7 * t)) * np.cos(4 * t),
                                             (1.0 + 0.46 * np.cos(7 * t)) * np.sin(4 * t)), 283),
    "dense 4:7 resonance trace with repeated winding", "physics", "winding")
add("phonon_polarization_loop", periodic(lambda t: (np.cos(t) + 0.22 * np.cos(4 * t),
                                                     0.72 * np.sin(t) + 0.16 * np.sin(3 * t)), 149),
    "anharmonic lattice-polarization trajectory", "physics", "orbit")
add("rayleigh_particle_orbit", periodic(lambda t: (1.25 * np.cos(t) + 0.12 * np.cos(2 * t),
                                                    0.58 * np.sin(t) - 0.08 * np.sin(2 * t)), 103),
    "elliptical surface-wave particle orbit with harmonic distortion", "physics", "orbit")
add("magnetic_island_chain", fourier_blob(((0.22, 6, 0.1), (0.08, 5, 1.0)), 181),
    "six-island perturbation around a magnetic flux surface", "physics", "field")
add("vortex_pair_streamline", cassini(149),
    "single Cassini oval representing a two-vortex streamline", "physics", "field")
add("charged_particle_drift", periodic(lambda t: (1.2 * np.cos(t) + 0.065 * np.cos(9 * t),
                                                  0.8 * np.sin(t) + 0.065 * np.sin(9 * t)), 197),
    "closed guiding-center drift decorated by fast gyromotion", "physics", "winding")
add("standing_wave_loop", periodic(lambda t: (np.cos(t) + 0.12 * np.cos(6 * t),
                                              np.sin(t) - 0.09 * np.sin(6 * t)), 191),
    "periodic standing-wave displacement loop", "physics", "winding")
add("coupled_oscillator_orbit", periodic(lambda t: (np.sin(3 * t) + 0.24 * np.sin(8 * t),
                                                    np.sin(4 * t + 0.45)), 263),
    "closed orbit of two coupled incommensurate-looking modes", "physics", "winding")
add("fourier_periodic_orbit", periodic(lambda t: (1.1 * np.cos(t) + 0.17 * np.cos(3 * t + 0.4)
                                                   + 0.05 * np.sin(7 * t),
                                                   0.85 * np.sin(t) - 0.10 * np.sin(4 * t)
                                                   + 0.035 * np.cos(9 * t)), 223),
    "irregular but exactly periodic Fourier orbit", "physics", "winding")

# 57-88: special mathematical curves and geometric constructions.
add("circle", polar(lambda t: np.ones_like(t), 41), "minimal convex cyclic reference", "mathematics", "geometry")
add("eccentric_ellipse", periodic(lambda t: (1.75 * np.cos(t), 0.62 * np.sin(t)), 59),
    "high-eccentricity ellipse", "mathematics", "geometry")
add("superellipse", superellipse(4.5, 1.35, 0.9, 101), "Lamé superellipse with rounded rectangular character", "mathematics", "geometry")
add("constant_width_curve", polar(lambda t: 1.0 + 0.12 * np.cos(3 * t) + 0.025 * np.cos(6 * t), 113),
    "smooth constant-width-inspired three-harmonic oval", "mathematics", "geometry")
add("rounded_triangle", fourier_blob(((0.17, 3, 0.0), (0.035, 6, math.pi)), 97),
    "rounded triangular harmonic domain", "mathematics", "geometry")
add("rounded_square", fourier_blob(((0.13, 4, 0.0), (0.025, 8, math.pi)), 101),
    "rounded square generated by polar harmonics", "mathematics", "geometry")
add("rounded_pentagon", fourier_blob(((0.11, 5, 0.0), (0.018, 10, math.pi)), 109),
    "rounded pentagonal harmonic domain", "mathematics", "geometry")
add("reuleaux_surrogate", fourier_blob(((0.145, 3, math.pi), (0.026, 6, 0.0), (0.009, 9, math.pi)), 127),
    "smooth Reuleaux-triangle surrogate", "mathematics", "geometry")
add("cassini_oval", cassini(131), "single-loop Cassini oval", "mathematics", "geometry")
add("limacon", polar(lambda t: 1.0 + 0.48 * np.cos(t), 121), "convex-to-dimpled transition in a Pascal limaçon", "mathematics", "geometry")
add("deep_dimple_limacon", polar(lambda t: 1.0 + 0.76 * np.cos(t), 139), "strongly dimpled regular limaçon", "mathematics", "geometry")
add("egg_curve", polar(lambda t: 1.0 + 0.24 * np.sin(t) - 0.055 * np.cos(2 * t), 107, y_scale=1.18),
    "asymmetric smooth egg curve", "mathematics", "geometry")
add("pear_curve", periodic(lambda t: ((1.0 + 0.28 * np.sin(t)) * np.cos(t),
                                      1.18 * (1.0 + 0.12 * np.sin(t)) * np.sin(t)), 119),
    "regular pear-shaped algebraic surrogate", "mathematics", "geometry")
add("smooth_heart", periodic(lambda t: (0.95 * np.sin(t) - 0.26 * np.sin(2 * t),
                                        0.85 * np.cos(t) - 0.23 * np.cos(2 * t) - 0.08 * np.cos(3 * t)), 151),
    "smooth heart curve without a cuspidal bottom", "mathematics", "geometry")
add("gerono_lemniscate", periodic(lambda t: (1.25 * np.sin(t), 0.36 * np.sin(2 * t)), 181),
    "lemniscate of Gerono with a transverse central crossing", "mathematics", "winding")
add("crossed_harmonic_oval", periodic(lambda t: (np.sin(t) + 0.20 * np.sin(3 * t),
                                                 0.50 * np.sin(2 * t) + 0.08 * np.sin(5 * t)), 191),
    "self-crossing harmonic oval with four curvature regimes", "mathematics", "winding")
add("folium_surrogate", periodic(lambda t: (np.sin(t) + 0.42 * np.sin(2 * t),
                                            0.72 * np.sin(2 * t) - 0.18 * np.sin(3 * t)), 197),
    "closed folium-inspired crossing curve", "mathematics", "winding")
add("epitrochoid_5_3", periodic(lambda t: ((1.0 + 0.29 * np.cos(5 * t)) * np.cos(t),
                                           (1.0 + 0.29 * np.cos(5 * t)) * np.sin(t)), 173),
    "five-lobed epitrochoid-style rim", "mathematics", "geometry")
add("hypotrochoid_7_3", periodic(lambda t: ((0.82 + 0.33 * np.cos(7 * t)) * np.cos(t),
                                            (0.82 + 0.33 * np.cos(7 * t)) * np.sin(t)), 211),
    "seven-lobed hypotrochoid-style rim", "mathematics", "geometry")
add("soft_deltoid", periodic(lambda t: (2 * np.cos(t) + 0.82 * np.cos(2 * t),
                                        2 * np.sin(t) - 0.82 * np.sin(2 * t)), 149),
    "regularized three-corner deltoid", "mathematics", "geometry")
add("soft_astroid", periodic(lambda t: (1.15 * np.cos(t) ** 3 + 0.08 * np.cos(t),
                                        1.15 * np.sin(t) ** 3 + 0.08 * np.sin(t)), 157),
    "regularized four-corner astroid", "mathematics", "geometry")
add("soft_nephroid", periodic(lambda t: (3 * np.cos(t) - 0.78 * np.cos(3 * t),
                                         3 * np.sin(t) - 0.78 * np.sin(3 * t)), 167),
    "regularized nephroid", "mathematics", "geometry")
add("butterfly_curve", periodic(lambda t: (np.sin(t) * (1.0 + 0.32 * np.cos(4 * t)),
                                           np.cos(t) * (0.72 + 0.25 * np.cos(4 * t))), 181),
    "four-wing harmonic butterfly curve", "mathematics", "winding")
add("four_petalled_rim", polar(lambda t: 1.0 + 0.38 * np.cos(4 * t), 157),
    "four-petal rose rim that avoids a central cusp", "mathematics", "geometry")
add("seven_petalled_rim", polar(lambda t: 1.0 + 0.31 * np.cos(7 * t), 191),
    "seven-petal radial rose rim", "mathematics", "geometry")
add("guilloche_loop", periodic(lambda t: ((1.0 + 0.22 * np.cos(11 * t)) * np.cos(2 * t),
                                          (1.0 + 0.22 * np.cos(11 * t)) * np.sin(2 * t)), 277),
    "two-winding eleven-frequency guilloché loop", "mathematics", "winding")
add("spirograph_9_4", periodic(lambda t: ((1.0 + 0.36 * np.cos(9 * t)) * np.cos(4 * t),
                                         (1.0 + 0.36 * np.cos(9 * t)) * np.sin(4 * t)), 293),
    "dense four-winding 9:4 spirograph", "mathematics", "winding")
add("maurer_rose_surrogate", periodic(lambda t: ((1.0 + 0.48 * np.cos(5 * t)) * np.cos(3 * t + 0.13),
                                                 (1.0 + 0.48 * np.cos(5 * t)) * np.sin(3 * t + 0.13)), 281),
    "multi-crossing Maurer-rose-inspired traversal", "mathematics", "winding")
add("harmonic_square", periodic(lambda t: (np.cos(t) + np.cos(3 * t) / 9.0 + np.cos(5 * t) / 25.0,
                                          np.sin(t) - np.sin(3 * t) / 9.0 + np.sin(5 * t) / 25.0), 127),
    "Fourier-rounded square", "mathematics", "geometry")
add("harmonic_pentagon", periodic(lambda t: (np.cos(t) + 0.11 * np.cos(4 * t) - 0.04 * np.cos(6 * t),
                                            np.sin(t) - 0.11 * np.sin(4 * t) - 0.04 * np.sin(6 * t)), 131),
    "Fourier-rounded pentagon", "mathematics", "geometry")
add("seeded_fourier_domain", periodic(lambda t: (np.cos(t) + 0.18 * np.cos(2 * t + 0.7)
                                                 + 0.09 * np.cos(5 * t - 0.4),
                                                 np.sin(t) + 0.14 * np.sin(3 * t + 0.2)
                                                 - 0.07 * np.sin(8 * t)), 173),
    "deterministic irregular Fourier domain", "mathematics", "geometry")
add("constant_width_seven", polar(lambda t: 1.0 + 0.065 * np.cos(7 * t) + 0.018 * np.cos(14 * t), 149),
    "subtle sevenfold constant-width-inspired curve", "mathematics", "geometry")

# 89-112: natural forms, illustrations and scientific outlines.
add("leaf_outline", periodic(lambda t: (1.35 * np.cos(t), 0.62 * np.sin(t) * (1.0 - 0.28 * np.cos(t))), 113),
    "asymmetric leaf with rounded stem and tip", "illustration", "profile")
add("ginkgo_leaf", polar(lambda t: 0.85 + 0.27 * np.cos(2 * t) - 0.12 * np.cos(4 * t), 137,
                         x_scale=1.25, y_scale=0.95),
    "fan-shaped ginkgo leaf", "illustration", "profile")
add("candle_flame", periodic(lambda t: (0.48 * np.sin(t) * (1 + 0.35 * np.cos(t)),
                                        -0.9 * np.cos(t) + 0.15 * np.cos(2 * t)), 127),
    "regular flame silhouette", "illustration", "profile")
add("storm_cloud", fourier_blob(((0.18, 5, 0.2), (0.08, 8, -0.6), (0.04, 13, 0.3)), 167),
    "irregular closed storm-cloud boundary", "illustration", "profile")
add("butterfly_silhouette", periodic(lambda t: (np.sin(t) * (1.0 + 0.42 * np.cos(4 * t)),
                                                np.cos(t) * (0.68 + 0.30 * np.cos(4 * t))
                                                + 0.09 * np.cos(2 * t)), 197),
    "closed butterfly silhouette with four distinct wings", "illustration", "profile")
add("gliding_bird", catmull_rom([(-1.8, 0.0), (-1.0, 0.35), (-0.35, 0.12), (0.0, 0.55),
                                 (0.35, 0.12), (1.0, 0.35), (1.8, 0.0), (0.65, -0.25),
                                 (0.0, -0.12), (-0.65, -0.25)], 6),
    "closed gliding-bird planform", "illustration", "profile")
add("bat_emblem", catmull_rom([(-1.8, 0.2), (-1.1, 0.55), (-0.8, 0.05), (-0.35, 0.4),
                               (0.0, 0.1), (0.35, 0.4), (0.8, 0.05), (1.1, 0.55),
                               (1.8, 0.2), (1.15, -0.25), (0.45, -0.15), (0.0, -0.6),
                               (-0.45, -0.15), (-1.15, -0.25)], 5),
    "symmetrical bat-wing emblem", "illustration", "profile")
add("cat_head", catmull_rom([(-1.0, -0.5), (-1.05, 0.45), (-0.8, 1.2), (-0.35, 0.82),
                             (0.35, 0.82), (0.8, 1.2), (1.05, 0.45), (1.0, -0.5),
                             (0.45, -1.0), (-0.45, -1.0)], 6),
    "rounded cat-head silhouette with two ears", "illustration", "profile")
add("owl_mask", fourier_blob(((0.18, 2, math.pi), (0.14, 4, 0.0), (0.05, 7, 0.4)), 143),
    "wide-eyed owl-mask outline", "illustration", "profile")
add("heraldic_shield", catmull_rom([(-1.0, 0.9), (0.0, 1.15), (1.0, 0.9), (0.85, -0.15),
                                    (0.0, -1.35), (-0.85, -0.15)], 7),
    "rounded heraldic shield", "illustration", "profile")
add("helmet_visor", periodic(lambda t: (1.35 * np.cos(t) + 0.18 * np.cos(2 * t),
                                        0.78 * np.sin(t) - 0.14 * np.sin(2 * t)), 109),
    "asymmetric protective-visor perimeter", "illustration", "profile")
add("guitar_body", catmull_rom([(-0.45, -1.3), (-0.9, -0.9), (-0.75, -0.25), (-0.38, 0.0),
                                (-0.68, 0.48), (-0.42, 1.05), (0.0, 1.2), (0.42, 1.05),
                                (0.68, 0.48), (0.38, 0.0), (0.75, -0.25), (0.9, -0.9),
                                (0.45, -1.3)], 5),
    "complete double-bout guitar-body outline", "illustration", "profile")
add("ceramic_vase", periodic(lambda t: ((0.62 + 0.20 * np.cos(2 * t) - 0.08 * np.cos(4 * t)) * np.cos(t),
                                        1.35 * np.sin(t)), 127),
    "closed vase silhouette with neck and belly", "illustration", "profile")
add("laboratory_bottle", catmull_rom([(-0.42, -1.2), (-0.78, -0.7), (-0.72, 0.25), (-0.35, 0.7),
                                      (-0.28, 1.25), (0.28, 1.25), (0.35, 0.7), (0.72, 0.25),
                                      (0.78, -0.7), (0.42, -1.2)], 6),
    "rounded laboratory reagent-bottle perimeter", "illustration", "profile")
add("ergonomic_mouse", periodic(lambda t: (1.15 * np.cos(t) + 0.12 * np.cos(2 * t),
                                           0.72 * np.sin(t) + 0.08 * np.sin(3 * t)), 101),
    "ergonomic computer-mouse planform", "illustration", "profile")
add("telephone_handset", catmull_rom([(-1.55, -0.45), (-1.05, -0.8), (-0.45, -0.55),
                                      (0.45, -0.55), (1.05, -0.8), (1.55, -0.45),
                                      (1.25, 0.25), (0.65, 0.45), (-0.65, 0.45), (-1.25, 0.25)], 6),
    "closed industrial handset envelope", "illustration", "profile")
add("volcanic_island", fourier_blob(((0.24, 3, 0.7), (0.12, 5, -0.4), (0.06, 11, 0.2)), 173),
    "island coastline with bays, capes and a sheltered side", "illustration", "map")
add("polar_ice_floe", fourier_blob(((0.12, 5, 0.4), (0.09, 8, -0.7), (0.045, 17, 0.2)), 191),
    "irregular polar ice-floe boundary", "illustration", "map")
add("crater_lake", fourier_blob(((0.08, 2, 0.2), (0.06, 7, -0.3), (0.025, 13, 0.8)), 149),
    "closed volcanic crater-lake shoreline", "illustration", "map")
add("hurricane_eyewall", periodic(lambda t: ((1.0 + 0.13 * np.cos(3 * t + 0.5)
                                              + 0.055 * np.sin(8 * t)) * np.cos(t + 0.12 * np.sin(2 * t)),
                                             (0.82 + 0.07 * np.cos(5 * t)) * np.sin(t)), 197),
    "asymmetric closed eyewall observed in a rotating storm", "illustration", "field")
add("cell_membrane", fourier_blob(((0.11, 6, 0.1), (0.07, 9, -0.5), (0.035, 15, 0.7)), 181),
    "deformed biological cell membrane", "illustration", "field")
add("red_blood_cell_projection", periodic(lambda t: ((1.25 + 0.13 * np.cos(2 * t)) * np.cos(t),
                                                     (0.72 - 0.16 * np.cos(2 * t)) * np.sin(t)), 127),
    "biconcave red-blood-cell projection", "illustration", "field")
add("pollen_grain", fourier_blob(((0.13, 12, 0.0), (0.035, 5, 0.7)), 193),
    "spined pollen-grain section", "illustration", "field")
add("seed_pod", periodic(lambda t: (1.45 * np.cos(t),
                                    0.48 * np.sin(t) * (1.0 + 0.32 * np.cos(3 * t))), 131),
    "tapered botanical seed-pod outline", "illustration", "profile")

# 113-128: closed adaptations of the earlier open and PH B-spline galleries.
add("closed_endurance_circuit", catmull_rom([(-2.3, -0.2), (-1.6, -1.0), (-0.35, -1.25),
                                             (0.55, -0.6), (1.45, -1.05), (2.45, -0.3),
                                             (2.2, 0.75), (1.15, 1.15), (0.1, 0.65),
                                             (-0.75, 1.3), (-1.85, 0.8)], 6),
    "closed reconstruction of the open endurance-circuit example", "adapted", "track")
add("closed_island_outline", fourier_blob(((0.23, 3, 0.8), (0.11, 5, 0.0)), 157),
    "exactly closed version of the earlier near-closed island", "adapted", "map")
add("closed_trefoil_projection", polar(lambda t: 1.0 + 0.30 * np.cos(3 * t), 151),
    "exactly closed trefoil-inspired outline", "adapted", "geometry")
add("closed_lissajous_2_3", periodic(lambda t: (np.sin(2 * t + 0.9), np.sin(3 * t)), 211),
    "exactly closed replacement for the open Lissajous 2:3 example", "adapted", "winding")
add("closed_lissajous_3_4", polar(lambda t: 1.0 + 0.12 * np.cos(3 * t) + 0.06 * np.cos(4 * t), 229),
    "exactly closed radial-harmonic version of the earlier 3:4 example", "adapted", "geometry")
add("closed_gear_wheel", polar(lambda t: 1.0 + 0.13 * smooth_square(np.sin(9 * t), 0.06), 241),
    "exactly closed nine-tooth gear from the open mechanical gallery", "adapted", "profile")
add("closed_cam_with_notch", polar(lambda t: 1.0 + 0.28 * np.cos(t)
                                   - 0.18 * np.exp(-((np.angle(np.exp(1j * (t - 4.1))) / 0.24) ** 2)), 211),
    "closed cam lobe retaining the original concave dwell notch", "adapted", "profile")
add("closed_spirograph_dense", periodic(lambda t: ((1.0 + 0.28 * np.cos(13 * t)) * np.cos(2 * t),
                                                   (1.0 + 0.28 * np.cos(13 * t)) * np.sin(2 * t)), 313),
    "closed two-winding version of the dense spirograph", "adapted", "winding")
add("closed_heart_outline", periodic(lambda t: (16 * np.sin(t) ** 3 / 17.0,
                                                (13 * np.cos(t) - 5 * np.cos(2 * t)
                                                 - 2 * np.cos(3 * t) - np.cos(4 * t)) / 17.0
                                                + 0.025 * np.sin(t)), 181),
    "closed regularized form of the former near-closed heart", "adapted", "profile")
add("closed_cloud_outline", fourier_blob(((0.18, 5, 0.0), (0.07, 9, 0.6)), 167),
    "closed version of the scalloped cloud illustration", "adapted", "profile")
add("closed_wave_washer", polar(lambda t: 1.0 + 0.09 * np.sin(6 * t), 149),
    "closed version of the PH B-spline wave-washer case", "adapted", "profile")
add("closed_zigzag_star", polar(lambda t: 1.0 + 0.46 * smooth_square(np.cos(12 * t), 0.10), 293),
    "closed twelve-fold zigzag star adapted from the G8 B-spline gallery", "adapted", "profile")
add("closed_oval_with_chicane", catmull_rom([(-2.3, 0.0), (-1.7, -0.85), (-0.3, -0.8),
                                             (0.4, -0.25), (0.95, -0.85), (2.0, -0.6),
                                             (2.4, 0.2), (1.65, 0.9), (0.4, 0.75),
                                             (-0.3, 0.2), (-0.95, 0.85), (-2.0, 0.65)], 6),
    "closed version of the oval-with-chicane route", "adapted", "track")
add("closed_epitrochoid", polar(lambda t: 1.0 + 0.22 * np.cos(5 * t), 163),
    "exactly closed five-lobe epitrochoid from the open gallery", "adapted", "geometry")
add("closed_hypotrochoid", polar(lambda t: 0.8 + 0.28 * np.cos(7 * t), 197),
    "exactly closed seven-lobe hypotrochoid from the open gallery", "adapted", "geometry")
add("closed_loop_garland", polar(lambda t: 1.0 + 0.28 * np.cos(6 * t) + 0.05 * np.sin(11 * t), 257,
                                 x_scale=1.35, y_scale=0.72),
    "periodic six-lobe garland inspired by the pathological open example", "adapted", "geometry")

assert len(CASES) == 128, f"expected 128 cases, found {len(CASES)}"
assert len({case.name for case in CASES}) == 128, "closed example names must be unique"


def seam_candidates(points: np.ndarray) -> list[int]:
    """Prefer a seam surrounded by strong, consistently oriented turns."""
    previous = points - np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0) - points
    cross = previous[:, 0] * following[:, 1] - previous[:, 1] * following[:, 0]
    scale = np.linalg.norm(previous, axis=1) * np.linalg.norm(following, axis=1)
    normalized = np.divide(np.abs(cross), scale, out=np.zeros_like(cross), where=scale > 0.0)
    signs = np.sign(cross)
    scores = normalized.copy()
    scores += 2.0 * (signs == np.roll(signs, 1))
    scores += 2.0 * (signs == np.roll(signs, -1))
    order = np.argsort(-scores, kind="stable").tolist()
    return [0] + [int(index) for index in order if index != 0]


def compile_case(case: Case) -> tuple[np.ndarray, CubicPHSplineClosed]:
    points = np.asarray(case.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise AssertionError(f"{case.name}: invalid point array {points.shape}")
    if not np.all(np.isfinite(points)):
        raise AssertionError(f"{case.name}: nonfinite input")
    last_seam_error: InterpolationDomainError | None = None
    for shift in seam_candidates(points):
        rotated = np.roll(points, -shift, axis=0)
        try:
            return rotated, CubicPHSplineClosed(rotated.tolist())
        except InterpolationDomainError as error:
            if error.quantity != "seam continuity":
                raise
            last_seam_error = error
    assert last_seam_error is not None
    raise last_seam_error


def verify(curve: CubicPHSplineClosed, points: np.ndarray) -> None:
    count = points.shape[0]
    assert curve.closed and curve.num_points == count and curve.degree == 3
    for index, expected in enumerate(points):
        assert np.array_equal(curve.point(index / count), expected)
    assert np.array_equal(curve.point(0.0), curve.point(1.0))
    for index, right in enumerate(curve._segments):
        left = curve._segments[index - 1]
        assert np.allclose(left.tangent_local(1.0), right.tangent_local(0.0), atol=1e-12)
        left_curvature = left.curvature_local(1.0)
        right_curvature = right.curvature_local(0.0)
        kind = curve._joint_kinds[index]
        if kind == "g2":
            curvature_scale = max(abs(left_curvature), abs(right_curvature), np.finfo(float).eps)
            assert abs(left_curvature - right_curvature) / curvature_scale < 1e-10
        elif kind == "inflection":
            assert left_curvature * right_curvature < 0.0
        else:
            assert kind == "transition"
            assert (left.chi == 0.0) ^ (right.chi == 0.0)
    assert curve._joint_kinds[0] == "g2"
    for fraction in (0.07, 0.31, 0.67, 0.93):
        target = fraction * curve.length
        recovered = curve.arc_length(curve.parameter_at_length(target))
        assert abs(recovered - target) <= 256.0 * np.finfo(float).eps * curve.length


def rendered_samples(curve: CubicPHSplineClosed, count: int = 1100) -> np.ndarray:
    return np.asarray([curve.point(float(u)) for u in np.linspace(0.0, 1.0, count)])


def draw_winding(ax: plt.Axes, samples: np.ndarray, color: str) -> None:
    segments = np.stack((samples[:-1], samples[1:]), axis=1)
    collection = LineCollection(segments, cmap="viridis", linewidth=2.05, zorder=3)
    collection.set_array(np.linspace(0.0, 1.0, len(segments)))
    ax.add_collection(collection)
    ax.plot([], [], color=color, linewidth=2.0, label="closed cubic PH · traversal hue")


def draw_vectors(ax: plt.Axes, curve: CubicPHSplineClosed, color: str) -> None:
    parameters = np.linspace(0.0, 1.0, 13, endpoint=False)
    locations = np.asarray([curve.point(float(u)) for u in parameters])
    vectors = np.asarray([curve.curvature_vector(float(u)) for u in parameters])
    norms = np.linalg.norm(vectors, axis=1)
    nonzero = norms > np.finfo(float).eps
    if not np.any(nonzero):
        return
    vectors[nonzero] /= norms[nonzero, None]
    extent = np.ptp(locations, axis=0)
    vectors *= 0.09 * max(float(np.max(extent)), 1.0)
    ax.quiver(locations[:, 0], locations[:, 1], vectors[:, 0], vectors[:, 1],
              angles="xy", scale_units="xy", scale=1.0, color=color, alpha=0.55,
              width=0.004, zorder=2)


def render(index: int, case: Case) -> None:
    points, curve = compile_case(case)
    verify(curve, points)
    samples = rendered_samples(curve)
    stations = np.asarray([
        curve.point_at_length(float(distance))
        for distance in np.linspace(0.0, curve.length, 19, endpoint=False)
    ])
    color = CATEGORY_COLORS[case.category]
    fig, ax = plt.subplots(figsize=(5.8, 4.7), dpi=105, facecolor=PAGE)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color=GRID, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)

    if case.style in {"profile", "geometry", "map"}:
        ax.fill(samples[:, 0], samples[:, 1], color=color, alpha=0.065, zorder=1)
    if case.style in {"orbit", "field", "metrology"}:
        ax.axhline(0.0, color=MUTED, linewidth=0.6, alpha=0.45)
        ax.axvline(0.0, color=MUTED, linewidth=0.6, alpha=0.45)
    if case.style == "winding":
        draw_winding(ax, samples, color)
    else:
        ax.plot(samples[:, 0], samples[:, 1], color=color, linewidth=2.05,
                label="closed cubic PH", zorder=3)
    if case.style == "field":
        draw_vectors(ax, curve, color)

    step = max(1, math.ceil(len(points) / 42))
    shown = points[::step]
    ax.scatter(shown[:, 0], shown[:, 1], s=10, facecolor=SURFACE, edgecolor=POINT,
               linewidth=0.7, zorder=4, label="interpolation nodes")
    ax.scatter(stations[:, 0], stations[:, 1], s=10, color=STATION, zorder=5,
               label="equal-distance stations")

    auxiliary = len(curve.aux_inflection_points)
    fig.suptitle(f"{index:03d} · {case.name.replace('_', ' ')}", x=0.06, ha="left",
                 color=INK, fontsize=10.5)
    ax.set_title(
        f"{case.category} · {curve.num_points} nodes · {len(curve._segments)} cubic spans · "
        f"G2 seam · {auxiliary} auxiliary inflections\n{case.note}",
        loc="left", color=INK, fontsize=7.15,
    )
    ax.legend(loc="best", frameon=False, fontsize=6.8)
    fig.savefig(OUT / f"{index:03d}_{case.name}.png", bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="verify without rendering")
    args = parser.parse_args()
    failures: list[tuple[int, str, str]] = []
    if not args.check_only:
        for path in OUT.glob("[0-9][0-9][0-9]_*.png"):
            path.unlink()
    for index, case in enumerate(CASES, 1):
        try:
            if args.check_only:
                points, curve = compile_case(case)
                verify(curve, points)
            else:
                render(index, case)
            print(f"ok {index:03d} {case.name}")
        except Exception as error:  # report the complete corpus, then fail
            failures.append((index, case.name, f"{type(error).__name__}: {error}"))
            print(f"FAIL {index:03d} {case.name}: {type(error).__name__}: {error}")
    if failures:
        raise SystemExit(f"{len(failures)} of 128 closed examples failed")
    action = "verified" if args.check_only else "rendered"
    print(f"{action} 128 diverse closed cubic-PH examples")


if __name__ == "__main__":
    main()
