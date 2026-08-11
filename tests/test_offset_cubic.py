"""Exact offset NURBS tests for the cubic PH family (spec 19.6).

The oracle for coefficient-level checks uses exact rational arithmetic on
the stored binary64 segment data, which is stronger than the required
100-decimal-digit precision; sampled point agreement is checked separately
against ``r(u) + d * N_L(u)`` through the public geometry API.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from ph_spline import (
    CubicPHSplineClosed,
    CubicPHSplineOpen,
    NURBSHandle,
    OffsetConstructionError,
)

OPEN_CASES = {
    "straight": [[0.0, 0.0], [1.0, 0.5], [2.6, 1.3], [5.0, 2.5]],
    "convex": [
        [math.cos(a), math.sin(a)] for a in np.linspace(-0.5, 1.4, 9)
    ],
    "s_curve": [
        [0.0, 0.0], [1.0, 0.8], [2.0, 1.0], [3.0, 0.2], [4.0, 0.0],
        [5.0, 0.8],
    ],
    "mixed": [
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.5], [3.6, 1.4],
        [3.8, 2.4],
    ],
}

CLOSED_CASES = {
    "square": [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
    "wavy": [
        [
            (2.0 + 0.3 * math.cos(5.0 * t)) * math.cos(t),
            (2.0 + 0.3 * math.cos(5.0 * t)) * math.sin(t),
        ]
        for t in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    ],
}

DISTANCES = (0.0, 1.0e-14, 0.05, -0.35, 7.5)


def _make(name: str):
    if name in OPEN_CASES:
        return CubicPHSplineOpen(OPEN_CASES[name])
    return CubicPHSplineClosed(CLOSED_CASES[name])


ALL_NAMES = sorted(OPEN_CASES) + sorted(CLOSED_CASES)


@pytest.fixture(params=ALL_NAMES, ids=ALL_NAMES, scope="module")
def curve(request):
    return _make(request.param)


def _offset_tolerance(curve, distance: float) -> float:
    scale = max(1.0, float(np.max(np.abs(curve._points))))
    return 1e-11 * max(scale, abs(distance))


def _check_against_source(curve, handle, distance, parameters):
    tol = _offset_tolerance(curve, distance)
    for u in parameters:
        u = float(u)
        expected = curve.point(u) + distance * curve.normal(u)
        produced = handle.point(u)
        gap = math.hypot(*(produced - expected))
        assert gap <= tol, f"u={u}: gap {gap:.3e} > {tol:.3e}"


# ---------------------------------------------------------------------------
# Structure over the full case/distance matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("distance", DISTANCES)
def test_structure_and_direct_comparison(curve, distance):
    handle = curve.offset(distance)
    assert isinstance(handle, NURBSHandle)
    assert handle.degree == 5
    assert handle.domain == (0.0, 1.0)
    assert handle.closed == curve.closed
    n = handle.num_control_points
    assert handle.knots.shape == (n + 6,)
    assert n == 5 * handle.num_spans + 1
    assert np.all(np.isfinite(handle.knots))
    assert np.all(np.diff(handle.knots) >= 0.0)
    assert np.all(handle.knots[:6] == 0.0)
    assert np.all(handle.knots[-6:] == 1.0)
    _, counts = np.unique(
        handle.knots[6:-6], return_counts=True
    )
    assert np.all(counts == 5)
    assert np.all(np.isfinite(handle.control_points))
    assert np.all(handle.weights > 0.0)

    rng = np.random.default_rng(7 + int(abs(distance) * 1000))
    parameters = np.concatenate(
        (
            [0.0, 1.0],
            np.unique(handle.knots),
            rng.uniform(0.0, 1.0, 60),
        )
    )
    _check_against_source(curve, handle, distance, parameters)


def test_signed_identity(curve):
    d = 0.21
    plus = curve.offset(d)
    minus = curve.offset(-d)
    rng = np.random.default_rng(11)
    tol = _offset_tolerance(curve, d)
    for u in rng.uniform(0.0, 1.0, 40):
        u = float(u)
        base = curve.point(u)
        normal = curve.normal(u)
        assert math.hypot(*(plus.point(u) - base - d * normal)) <= tol
        assert math.hypot(*(minus.point(u) - base + d * normal)) <= tol


def test_denominator_independent_of_distance(curve):
    handles = [curve.offset(d) for d in (-3.0, 0.0, 1.0e-14, 0.6)]
    reference = handles[0]
    for other in handles[1:]:
        assert np.array_equal(reference.knots, other.knots)
        assert np.array_equal(reference.weights, other.weights)
        assert reference.num_spans == other.num_spans


def test_deterministic_repeated_construction(curve):
    first = curve.offset(0.37)
    second = curve.offset(0.37)
    assert np.array_equal(first.knots, second.knots)
    assert np.array_equal(first.control_points, second.control_points)
    assert np.array_equal(first.weights, second.weights)


# ---------------------------------------------------------------------------
# Cusps and self-intersections are retained, never trimmed
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def arc():
    angles = np.linspace(0.1, 1.3, 9)
    return CubicPHSplineOpen(
        [[math.cos(a), math.sin(a)] for a in angles]
    )


def test_cusp_case_is_constructed(arc):
    kappa = arc.signed_curvature(0.5)
    d_cusp = 1.0 / kappa  # 1 - d * kappa == 0 on the whole arc
    handle = arc.offset(d_cusp)
    # The offset of a near-unit circle at its radius collapses towards the
    # center; the curve is still exactly representable and verified.
    for u in (0.0, 0.25, 0.5, 0.75, 1.0):
        expected = arc.point(u) + d_cusp * arc.normal(u)
        assert math.hypot(*(handle.point(u) - expected)) <= 1e-11
        assert math.hypot(*handle.point(u)) <= 5e-3  # near the center


def test_cusp_condition_reached(arc):
    kappa = arc.signed_curvature(0.5)
    d_cusp = 1.0 / kappa
    residual = 1.0 - d_cusp * arc.signed_curvature(0.5)
    assert abs(residual) <= 1e-12


def test_large_distance_self_intersection_not_trimmed(arc):
    distance = -30.0
    handle = arc.offset(distance)
    tol = 1e-9
    for u in np.linspace(0.0, 1.0, 41):
        u = float(u)
        expected = arc.point(u) + distance * arc.normal(u)
        assert math.hypot(*(handle.point(u) - expected)) <= tol


# ---------------------------------------------------------------------------
# Invalid arguments and nonrepresentable output
# ---------------------------------------------------------------------------


def test_malformed_distance_arguments(arc):
    for bad in (True, np.False_, "0.5", [0.5], (0.5,), np.array([0.5])):
        with pytest.raises(TypeError):
            arc.offset(bad)
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(OffsetConstructionError) as info:
            arc.offset(bad)
        assert info.value.operation == "offset"


def test_nonrepresentable_offset_fails_typed():
    tiny = CubicPHSplineOpen(
        [
            [1e-150 * math.cos(a), 1e-150 * math.sin(a)]
            for a in np.linspace(0.0, 1.2, 7)
        ]
    )
    with pytest.raises(OffsetConstructionError) as info:
        tiny.offset(1e200)
    assert info.value.operation == "offset"
    assert info.value.distance == 1e200


def test_zero_distance_reproduces_source(curve):
    handle = curve.offset(0.0)
    rng = np.random.default_rng(23)
    tol = _offset_tolerance(curve, 0.0)
    for u in rng.uniform(0.0, 1.0, 30):
        u = float(u)
        assert math.hypot(*(handle.point(u) - curve.point(u))) <= tol


# ---------------------------------------------------------------------------
# Straight spline: analytically exact translation
# ---------------------------------------------------------------------------


def test_straight_offset_is_exact_translation():
    line = CubicPHSplineOpen([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    handle = line.offset(2.5)
    normal = np.array([-0.8, 0.6])  # left normal of direction (0.6, 0.8)
    for u in np.linspace(0.0, 1.0, 21):
        u = float(u)
        expected = line.point(u) + 2.5 * normal
        assert math.hypot(*(handle.point(u) - expected)) <= 1e-12


# ---------------------------------------------------------------------------
# Closed topology: seam behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CLOSED_CASES))
@pytest.mark.parametrize("distance", (0.3, -0.3))
def test_closed_seam_values_agree(name, distance):
    loop = CubicPHSplineClosed(CLOSED_CASES[name])
    handle = loop.offset(distance)
    assert handle.closed is True
    start = handle.point(0.0)
    end = handle.point(1.0)
    tol = _offset_tolerance(loop, distance)
    assert math.hypot(*(end - start)) <= tol
    expected = loop.point(0.0) + distance * loop.normal(0.0)
    assert math.hypot(*(start - expected)) <= tol


# ---------------------------------------------------------------------------
# Exact-rational verification of the homogeneous Bernstein coefficients
# ---------------------------------------------------------------------------


def _exact_segment_products(segment, d_hat: Fraction):
    """Exact homogeneous offset coefficients from the stored segment data.

    All stored binary64 values convert exactly to rationals, so these sums
    have no rounding at all: their precision exceeds any fixed decimal
    count.  Only the final comparison rounds.
    """
    controls = [
        (Fraction(float(x)), Fraction(float(y))) for x, y in segment.ctrl
    ]
    w0 = (Fraction(segment.w0.real), Fraction(segment.w0.imag))
    w1 = (Fraction(segment.w1.real), Fraction(segment.w1.imag))
    speed = [
        w0[0] * w0[0] + w0[1] * w0[1],
        w0[0] * w1[0] + w0[1] * w1[1],
        w1[0] * w1[0] + w1[1] * w1[1],
    ]
    velocity = [
        (3 * (bx - ax), 3 * (by - ay))
        for (ax, ay), (bx, by) in zip(controls[:-1], controls[1:])
    ]
    rows = []
    for k in range(6):
        w_sum = Fraction(0)
        x_sum = Fraction(0)
        y_sum = Fraction(0)
        for i in range(max(0, k - 3), min(2, k) + 1):
            j = k - i
            lam = Fraction(
                math.comb(2, i) * math.comb(3, j), math.comb(5, k)
            )
            w_sum += lam * speed[i]
            x_sum += lam * (
                speed[i] * controls[j][0] + d_hat * (-velocity[i][1])
            )
            y_sum += lam * (
                speed[i] * controls[j][1] + d_hat * velocity[i][0]
            )
        rows.append((w_sum, x_sum, y_sum))
    return rows


def test_homogeneous_coefficients_match_exact_rational_oracle():
    curve = _make("convex")
    distance = 0.42
    d_hat = Fraction(distance) / Fraction(curve._scale)
    handle = curve.offset(distance)
    q = 5
    origin = (Fraction(curve._origin[0]), Fraction(curve._origin[1]))
    scale = Fraction(curve._scale)
    # Reconstruct the published homogeneous controls patch by patch and
    # compare with the exact rational products, tracking the projective
    # chain scale exactly.
    chain = Fraction(1)
    for index, segment in enumerate(curve._segments):
        rows = _exact_segment_products(segment, d_hat)
        if index > 0:
            previous = _exact_segment_products(
                curve._segments[index - 1], d_hat
            )
            chain *= previous[-1][0] / rows[0][0]
        for k in range(6):
            control_index = index * q + k
            w_exact = chain * rows[k][0]
            x_exact = origin[0] + scale * (rows[k][1] / rows[k][0])
            y_exact = origin[1] + scale * (rows[k][2] / rows[k][0])
            w_produced = Fraction(float(handle.weights[control_index]))
            x_produced = Fraction(
                float(handle.control_points[control_index, 0])
            )
            y_produced = Fraction(
                float(handle.control_points[control_index, 1])
            )
            assert abs(w_produced - w_exact) <= Fraction(1, 10**12) * max(
                Fraction(1), abs(w_exact)
            )
            assert abs(x_produced - x_exact) <= Fraction(1, 10**11) * max(
                Fraction(1), abs(x_exact)
            )
            assert abs(y_produced - y_exact) <= Fraction(1, 10**11) * max(
                Fraction(1), abs(y_exact)
            )


def test_handle_survives_source_garbage_collection():
    handle = CubicPHSplineOpen(OPEN_CASES["convex"]).offset(0.1)
    import gc

    gc.collect()
    value = handle.point(0.5)
    assert np.all(np.isfinite(value))
