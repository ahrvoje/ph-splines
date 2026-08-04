"""Generate 32 examples engineered to sit just inside the failure bounds.

Run:  python examples/generate_near_break.py
Output: examples/plots_near_break/NN_name.png

Each case pushes one (or several) of the documented limits -- the
turn-pair uniqueness bound, the chord-ratio floor, the collinearity
threshold, the reversal threshold, coordinate range, curvature range or
system size -- while staying on the constructible side.  Every case must
fully construct, pass the independent invariant check, and render;
the annotation states the margin to the bound that was approached.
"""

from __future__ import annotations

import math
import os

import numpy as np
from _common import (
    circle_points,
    log_spiral_points,
    plot_example,
    polyline_from_turns,
)

from cubic_ph_spline._constants import (
    CHORD_RATIO_MIN,
    COLLINEAR_EPS,
    THETA_UNIQUE,
)

OUT = os.path.join(os.path.dirname(__file__), "plots_near_break")

CASES: list[tuple[str, list, str]] = []


def add(name: str, points: list, note: str) -> None:
    CASES.append((name, points, note))


def pair_case(turn: float) -> tuple[list, str]:
    pts = polyline_from_turns([1.0, 1.0, 1.0], [turn, turn])
    s = 2.0 * turn
    margin = THETA_UNIQUE - s
    return pts, (
        f"adjacent turn sum {s:.6f} rad of bound {THETA_UNIQUE:.6f} "
        f"({100.0 * s / THETA_UNIQUE:.4f}%, margin {margin:.2e} rad)"
    )


def ratio_case(decades: float, n: int, turn: float) -> tuple[list, str]:
    lengths = list(10.0 ** np.linspace(-decades, 0.0, n))
    pts = polyline_from_turns(lengths, [turn] * (n - 1))
    ratio = 10.0 ** (-decades)
    return pts, (
        f"chord ratio {ratio:.1e} vs floor {CHORD_RATIO_MIN:.2e} "
        f"({ratio / CHORD_RATIO_MIN:.2g}x above the floor)"
    )


# 1-3: adjacent turn sums creeping up on the uniqueness bound ---------------

for turn, tag in [(2.046, "a"), (2.0482, "b"), (2.04843, "c")]:
    pts, note = pair_case(turn)
    add(f"pair_sum_bound_{tag}", pts, note)

# 4-5: single turns approaching pi (reversal) -------------------------------

for delta, tag in [(1e-4, "1e-4"), (1e-6, "1e-6")]:
    turn = math.pi - delta
    pts = polyline_from_turns([1.0, 1.0, 1.0], [turn, 0.3])
    add(
        f"turn_pi_minus_{tag}",
        pts,
        f"turn = pi - {delta:.0e} = {turn:.8f} rad; "
        f"sin(turn) = {math.sin(turn):.2e} vs zero-turn threshold {COLLINEAR_EPS:.2e}",
    )

# 6-8: chord-length ratios approaching the representability floor -----------

pts, note = ratio_case(10.0, 11, 0.22)
add("chord_ratio_1e-10", pts, note)
pts, note = ratio_case(11.0, 12, 0.20)
add("chord_ratio_1e-11", pts, note)
lengths = [5e-13, 1.0, 1.0, 1.0]
pts = polyline_from_turns(lengths, [0.4, 0.4, 0.4])
add(
    "chord_ratio_2x_floor",
    pts,
    f"shortest/longest chord = 5.0e-13, only {5e-13 / CHORD_RATIO_MIN:.2f}x "
    f"above the floor {CHORD_RATIO_MIN:.2e}",
)

# 9-10: curvature magnitude extremes ----------------------------------------

add(
    "curvature_1e-12",
    circle_points(R=1e12, a0=0.6, a1=0.6 + 8e-4, n=6),
    "circle radius 1e12: |kappa| = 1e-12 everywhere",
)
add(
    "curvature_1e12",
    circle_points(R=1e-12, a0=0.3, a1=1.6, n=8),
    "circle radius 1e-12: |kappa| = 1e12 everywhere",
)

# 11-14: coordinate range extremes ------------------------------------------

add(
    "coords_1e150",
    [[1e150 * p[0], 1e150 * p[1]] for p in circle_points(R=1.0, a0=0.2, a1=1.6, n=8)],
    "coordinate magnitude 1e150",
)
add(
    "coords_1e-150",
    [[1e-150 * p[0], 1e-150 * p[1]] for p in circle_points(R=1.0, a0=0.2, a1=1.6, n=8)],
    "coordinate magnitude 1e-150",
)
add(
    "coords_1e307",
    [[1e307 * p[0], 1e307 * p[1]] for p in circle_points(R=1.0, a0=0.4, a1=1.3, n=6)],
    "coordinates within one decade of the binary64 maximum",
)
add(
    "offset_1e15",
    circle_points(R=1e4, a0=0.3, a1=1.5, n=8, center=(1e15, -1e15)),
    "arc of size 1e4 offset 1e15 from the origin (coordinate ulp is 0.125 there)",
)

# 15-16: turns barely above the collinearity threshold ----------------------

for turn, tag in [(1e-9, "1e-9"), (1e-11, "1e-11")]:
    pts = polyline_from_turns([1.0] * 6, [turn] * 5)
    add(
        f"turns_{tag}",
        pts,
        f"every turn {turn:.0e} rad, {turn / COLLINEAR_EPS:.2g}x above the "
        f"numerically-zero threshold {COLLINEAR_EPS:.2e}",
    )

# 17-19: spacing and asymmetry stress ---------------------------------------

add(
    "alternating_asymmetric_turns",
    polyline_from_turns([1.0, 0.8, 0.8, 1.0], [1.9, 0.25, 1.9]),
    "turns alternate 1.9 / 0.25 / 1.9 rad: strongly asymmetric wedges",
)
add(
    "adjacent_chord_jump_1e5",
    polyline_from_turns([1.0, 1.0, 1e-5, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]),
    "interior chord 1e5 times shorter than its neighbors -- the largest "
    "jump whose residual still verifies below the strict 1e-11 gate "
    "(at 1e6 the residual's conditioning floor ~2e-11 is refused)",
)
add(
    "geometric_spacing_r10",
    polyline_from_turns([10.0 ** (-k) for k in range(12)], [0.35] * 11),
    "each chord one-tenth of the previous (total ratio 1e-11)",
)

# 20: boundary clamp exercised ----------------------------------------------

add(
    "boundary_clamp_engaged",
    polyline_from_turns([1.0, 1.0, 1.0, 1.0], [2.95, 0.5, 0.5]),
    "raw circumcircle start tangent exceeds the uniqueness clamp "
    "(clamped by the constructor, flag recorded)",
)

# 21-22: system size --------------------------------------------------------

add(
    "spiral_total_turn_20pi",
    log_spiral_points(a=0.2, b=0.028, t0=0.0, t1=20 * math.pi, n=320),
    "ten full revolutions, 319 segments, whorl gap ~19%",
)
add(
    "circle_1500pts",
    circle_points(R=1.0, a0=0.0, a1=2.8, n=1500),
    "1499 segments; 1498 nonlinear unknowns through the sparse solver",
)

# 23-25: solver stress via strong asymmetry ---------------------------------

add(
    "lambda_asymmetry",
    polyline_from_turns([1.0, 1.0, 1.0, 1.0], [0.05, 2.8, 0.05]),
    "turn pattern 0.05 / 2.8 / 0.05 rad drives extreme Bezier edge ratios",
)
add(
    "deep_speed_dip",
    polyline_from_turns([1.0, 0.3, 0.35], [1.75, 1.75]),
    "middle-segment tangent turn near 3.5 rad; speed dips deeply mid-segment",
)
add(
    "tiny_first_chord_high_turns",
    polyline_from_turns([1e-9, 1.0, 1.0], [1.95, 1.95]),
    "1e-9 first chord combined with turn pair sum 3.9 of 4.097",
)

# 26-28: perturbation and near-degeneracy -----------------------------------

rng = np.random.default_rng(20260804)
noisy = [
    [p[0] + float(dx), p[1] + float(dy)]
    for p, (dx, dy) in zip(
        circle_points(R=1.0, a0=0.0, a1=1.5, n=30),
        1e-13 * rng.standard_normal((30, 2)),
    )
]
add(
    "noisy_circle_1e-13",
    noisy,
    "circle samples perturbed by 1e-13 absolute noise (turns ~0.05 rad)",
)
add(
    "near_duplicate_points",
    polyline_from_turns([4e-13, 1.0, 1.0], [0.45, 0.45]),
    f"two points 4e-13 apart: ratio {4e-13:.1e} vs floor {CHORD_RATIO_MIN:.2e}",
)
add(
    "reversal_pi_minus_1e-8",
    polyline_from_turns([1.0, 1.0, 1.0], [math.pi - 1e-8, 0.25]),
    f"turn = pi - 1e-8; sin = 1e-8, {1e-8 / COLLINEAR_EPS:.0f}x above the "
    "zero-turn threshold",
)

# 29-31: combined stressors -------------------------------------------------

add(
    "reversal_plus_pair_bound",
    polyline_from_turns([1.0, 1.0, 1.0], [math.pi - 1e-3, 0.95]),
    f"pair sum {math.pi - 1e-3 + 0.95:.5f} of {THETA_UNIQUE:.5f} with a near-pi member",
)
add(
    "curvature_range_1e10",
    log_spiral_points(a=1.0, b=-0.35, t0=0.0, t1=21.5 * math.pi / 3.0, n=120),
    "inward spiral: |kappa| spans ~10 orders of magnitude in one spline",
)
add(
    "minimal_near_straight",
    polyline_from_turns([1.0, 1.0, 1.0], [1e-10, 1e-10]),
    "smallest curved system (2 unknowns) at turns 1e-10 rad",
)

# 32: everything at once ----------------------------------------------------

kitchen = [
    [
        1e10
        + (0.3 * math.exp(0.09 * t) * (1.0 + 0.05 * math.sin(0.5 * t))) * math.cos(t),
        -1e10
        + (0.3 * math.exp(0.09 * t) * (1.0 + 0.05 * math.sin(0.5 * t))) * math.sin(t),
    ]
    for t in np.linspace(0.0, 10 * math.pi, 500)
]
add(
    "kitchen_sink_500pts",
    kitchen,
    "500 points, radius-modulated spiral, five revolutions, offset 1e10",
)


def main() -> None:
    assert len(CASES) == 32, f"expected 32 cases, have {len(CASES)}"
    failures = []
    for i, (name, pts, note) in enumerate(CASES, start=1):
        try:
            curve = plot_example(i, name, pts, OUT, note=note)
            extra = ""
            if name == "boundary_clamp_engaged":
                assert curve._boundary_clamped[0], "expected start clamp"
                extra = " [start clamp engaged]"
            print(f"ok  {i:02d} {name:30s} {curve!r}{extra}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((i, name, exc))
            print(f"FAIL {i:02d} {name:30s} {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit(f"{len(failures)} case(s) failed")
    print(f"\nAll 32 near-break cases constructed and rendered to {OUT}")


if __name__ == "__main__":
    main()
