"""Exact offset NURBS tests for the PH B-spline family (spec 22.11)."""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from ph_spline import (
    DiscontinuousDerivativeError,
    NURBSHandle,
    OffsetConstructionError,
    PHBSplineClosed,
    PHBSplineOpen,
)
from ph_spline.bspline_segment import compile_span
from ph_spline.bspline_types import InversePolicy

OPEN_POINTS = [
    [0.0, 0.0], [1.0, 0.4], [2.0, -0.7], [3.0, 1.1], [2.2, 2.0],
]
CLOSED_POINTS = [
    [1.0, 0.0], [0.31, 0.95], [-0.81, 0.59], [-0.81, -0.59], [0.31, -0.95],
]
DOUBLE_LOOP = [
    [
        (2.0 + math.cos(0.5 * t)) * math.cos(t),
        (2.0 + math.cos(0.5 * t)) * math.sin(t),
    ]
    for t in np.linspace(0.0, 4.0 * math.pi, 16, endpoint=False)
]

DISTANCES = (0.0, 1.0e-13, 0.125, -0.4, 5.0)


def _build(name: str):
    if name == "open_g2":
        return PHBSplineOpen(OPEN_POINTS)
    if name == "open_g4":
        return PHBSplineOpen(OPEN_POINTS, g_order=4)
    if name == "open_g8":
        return PHBSplineOpen(OPEN_POINTS, g_order=8)
    if name == "closed_antiperiodic":
        return PHBSplineClosed(CLOSED_POINTS)
    if name == "closed_g4":
        return PHBSplineClosed(CLOSED_POINTS, g_order=4)
    if name == "closed_g8":
        return PHBSplineClosed(CLOSED_POINTS, g_order=8)
    if name == "closed_periodic":
        return PHBSplineClosed(DOUBLE_LOOP)
    raise AssertionError(name)


CASE_NAMES = [
    "open_g2",
    "open_g4",
    "open_g8",
    "closed_antiperiodic",
    "closed_g4",
    "closed_g8",
    "closed_periodic",
]


@pytest.fixture(params=CASE_NAMES, ids=CASE_NAMES, scope="module")
def curve(request):
    return _build(request.param)


def _tolerance(curve, distance: float) -> float:
    scale = max(1.0, float(np.max(np.abs(curve.points))))
    degree_factor = max(1.0, curve.preimage_degree**2 / 4.0)
    return 1e-11 * degree_factor * max(scale, abs(distance))


def _greville(handle: NURBSHandle) -> np.ndarray:
    q = handle.degree
    knots = handle.knots
    return np.array(
        [
            float(np.mean(knots[i + 1 : i + q + 1]))
            for i in range(handle.num_control_points)
        ]
    )


# ---------------------------------------------------------------------------
# Structure and direct comparison over the case/distance matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("distance", DISTANCES)
def test_structure_and_direct_comparison(curve, distance):
    handle = curve.offset(distance)
    assert isinstance(handle, NURBSHandle)
    q = 4 * curve.preimage_degree + 1
    assert handle.degree == q
    assert handle.domain == (0.0, 1.0)
    assert handle.closed == curve.closed
    n = handle.num_control_points
    assert handle.knots.shape == (n + q + 1,)
    assert n == handle.num_spans * q + 1
    assert np.all(np.isfinite(handle.knots))
    assert np.all(np.diff(handle.knots) >= 0.0)
    assert np.all(handle.knots[: q + 1] == 0.0)
    assert np.all(handle.knots[-(q + 1) :] == 1.0)
    _, counts = np.unique(handle.knots[q + 1 : -(q + 1)], return_counts=True)
    assert counts.size == 0 or np.all(counts == q)
    assert np.all(np.isfinite(handle.control_points))
    assert np.all(handle.weights > 0.0)

    tol = _tolerance(curve, distance)
    rng = np.random.default_rng(int(abs(distance) * 997) + 3)
    parameters = np.concatenate(
        (
            [0.0, 1.0],
            np.unique(handle.knots),
            np.clip(_greville(handle), 0.0, 1.0),
            rng.uniform(0.0, 1.0, 40),
        )
    )
    for u in parameters:
        u = float(u)
        expected = curve.point(u) + distance * curve.normal(u)
        gap = math.hypot(*(handle.point(u) - expected))
        assert gap <= tol, f"u={u}: gap {gap:.3e} > {tol:.3e}"


def test_signed_identity(curve):
    d = 0.19
    plus = curve.offset(d)
    minus = curve.offset(-d)
    tol = _tolerance(curve, d)
    for u in np.linspace(0.03, 0.97, 21):
        u = float(u)
        base = curve.point(u)
        normal = curve.normal(u)
        assert math.hypot(*(plus.point(u) - base - d * normal)) <= tol
        assert math.hypot(*(minus.point(u) - base + d * normal)) <= tol


def test_denominator_and_topology_independent_of_distance(curve):
    handles = [curve.offset(d) for d in DISTANCES]
    reference = handles[0]
    for other in handles[1:]:
        assert other.degree == reference.degree
        assert np.array_equal(reference.knots, other.knots)
        assert np.array_equal(reference.weights, other.weights)


def test_homogeneous_numerators_affine_in_distance(curve):
    handles = {d: curve.offset(d) for d in (0.0, 1.0, 2.0)}
    weights = handles[0.0].weights
    numerators = {
        d: handles[d].control_points * weights[:, None] for d in handles
    }
    midpoint = 0.5 * (numerators[0.0] + numerators[2.0])
    scale = float(np.max(np.abs(numerators[2.0]))) + 1.0
    assert np.max(np.abs(numerators[1.0] - midpoint)) <= 1e-11 * scale


def test_deterministic_repeated_construction(curve):
    first = curve.offset(-0.27)
    second = curve.offset(-0.27)
    assert np.array_equal(first.knots, second.knots)
    assert np.array_equal(first.control_points, second.control_points)
    assert np.array_equal(first.weights, second.weights)


def test_closed_seam_values(curve):
    if not curve.closed:
        pytest.skip("open topology")
    for d in (0.2, -0.2):
        handle = curve.offset(d)
        tol = _tolerance(curve, d)
        assert math.hypot(*(handle.point(0.0) - handle.point(1.0))) <= tol


# ---------------------------------------------------------------------------
# Snapshots and atomic version capture
# ---------------------------------------------------------------------------


def test_snapshot_exposes_offset():
    curve = PHBSplineOpen(OPEN_POINTS)
    snapshot = curve.snapshot()
    from_snapshot = snapshot.offset(0.2)
    from_curve = curve.offset(0.2)
    assert np.array_equal(
        from_snapshot.control_points, from_curve.control_points
    )
    assert np.array_equal(from_snapshot.weights, from_curve.weights)


def test_snapshot_offset_ignores_later_edits():
    curve = PHBSplineOpen(OPEN_POINTS)
    snapshot = curve.snapshot()
    reference = snapshot.offset(0.2)
    curve.move_point(2, [2.2, -0.5])
    repeated = snapshot.offset(0.2)
    assert np.array_equal(reference.control_points, repeated.control_points)
    assert np.array_equal(reference.weights, repeated.weights)
    assert np.array_equal(reference.knots, repeated.knots)


def test_offset_handle_is_atomic_under_edits():
    curve = PHBSplineOpen(OPEN_POINTS)
    handle = curve.offset(0.15)
    control_copy = np.array(handle.control_points, copy=True)
    weight_copy = np.array(handle.weights, copy=True)
    knot_copy = np.array(handle.knots, copy=True)
    value_before = handle.point(0.4)
    curve.move_point(2, [2.3, -0.4])
    curve.insert_point(3, [2.7, 0.4])
    assert np.array_equal(handle.control_points, control_copy)
    assert np.array_equal(handle.weights, weight_copy)
    assert np.array_equal(handle.knots, knot_copy)
    assert np.array_equal(handle.point(0.4), value_before)
    fresh = curve.offset(0.15)
    assert fresh.num_control_points != handle.num_control_points or (
        not np.array_equal(fresh.control_points, handle.control_points)
    )


# ---------------------------------------------------------------------------
# Tangent-discontinuity guard
# ---------------------------------------------------------------------------


def _kinked_curve():
    """A position-continuous, tangent-kinked two-span doctored source.

    The public constructor cannot produce a G0-only kink, so the guard is
    exercised by substituting a synthetic compiled state; ``offset`` reads
    exactly one captured state, which makes this substitution well posed.
    """
    curve = PHBSplineOpen([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    policy = InversePolicy()
    straight = np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j])
    turned_root = complex(math.cos(math.pi / 8.0), math.sin(math.pi / 8.0))
    turned = np.array([turned_root, turned_root, turned_root])
    jets = lambda w: np.array([w, 0.0j])  # noqa: E731
    first = compile_span(
        span_id=0,
        parameter_width=0.5,
        preimage=straight,
        preimage_left_jet=jets(straight[0]),
        preimage_right_jet=jets(straight[-1]),
        start=0.0j,
        regularity_lower=1.0,
        regularity_upper=1.0,
        inverse_policy=policy,
    )
    second = compile_span(
        span_id=1,
        parameter_width=0.5,
        preimage=turned,
        preimage_left_jet=jets(turned[0]),
        preimage_right_jet=jets(turned[-1]),
        start=complex(first.position[-1]),
        regularity_lower=1.0,
        regularity_upper=1.0,
        inverse_policy=policy,
    )
    state = replace(
        curve._state,
        spans=(first, second),
        span_knots=np.array([0.0, 0.5, 1.0]),
    )
    curve._state = state
    return curve


def test_nonzero_offset_rejected_at_tangent_kink():
    curve = _kinked_curve()
    with pytest.raises(DiscontinuousDerivativeError) as info:
        curve.offset(0.1)
    assert info.value.index == 1


def test_zero_offset_accepted_at_tangent_kink():
    curve = _kinked_curve()
    handle = curve.offset(0.0)
    assert handle.degree == 4 * curve.preimage_degree + 1
    assert np.all(handle.weights > 0.0)


# ---------------------------------------------------------------------------
# Invalid arguments
# ---------------------------------------------------------------------------


def test_malformed_distance_arguments():
    curve = PHBSplineOpen(OPEN_POINTS)
    for bad in (True, np.True_, "1", [1.0], np.array([1.0, 2.0]), None):
        with pytest.raises(TypeError):
            curve.offset(bad)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(OffsetConstructionError) as info:
            curve.offset(bad)
        assert info.value.operation == "offset"


def test_nonrepresentable_offset_fails_typed():
    scaled = PHBSplineOpen((1e-150 * np.asarray(OPEN_POINTS)).tolist())
    with pytest.raises(OffsetConstructionError):
        scaled.offset(1e250)


# ---------------------------------------------------------------------------
# Exact-rational verification of the homogeneous coefficients
# ---------------------------------------------------------------------------


def test_homogeneous_coefficients_match_exact_rational_oracle():
    curve = PHBSplineOpen(OPEN_POINTS)
    state = curve._state
    distance = 0.31
    d_hat = Fraction(distance) / Fraction(float(state.scale))
    handle = curve.offset(distance)
    q = handle.degree
    p = 2 * curve.preimage_degree + 1
    origin = (
        Fraction(float(state.origin[0])),
        Fraction(float(state.origin[1])),
    )
    scale = Fraction(float(state.scale))
    chain = Fraction(1)
    previous_rows = None
    for index, span in enumerate(state.spans):
        controls = [
            (Fraction(float(c.real)), Fraction(float(c.imag)))
            for c in span.position
        ]
        speed = [Fraction(float(s)) for s in span.speed]
        velocity = [
            (p * (bx - ax), p * (by - ay))
            for (ax, ay), (bx, by) in zip(controls[:-1], controls[1:])
        ]
        rows = []
        for k in range(q + 1):
            w_sum = Fraction(0)
            x_sum = Fraction(0)
            y_sum = Fraction(0)
            for i in range(max(0, k - p), min(p - 1, k) + 1):
                j = k - i
                lam = Fraction(
                    math.comb(p - 1, i) * math.comb(p, j), math.comb(q, k)
                )
                w_sum += lam * speed[i]
                x_sum += lam * (
                    speed[i] * controls[j][0] + d_hat * (-velocity[i][1])
                )
                y_sum += lam * (
                    speed[i] * controls[j][1] + d_hat * velocity[i][0]
                )
            rows.append((w_sum, x_sum, y_sum))
        if previous_rows is not None:
            chain *= previous_rows[-1][0] / rows[0][0]
        for k in range(q + 1):
            control_index = index * q + k
            w_exact = chain * rows[k][0]
            x_exact = origin[0] + scale * (rows[k][1] / rows[k][0])
            y_exact = origin[1] + scale * (rows[k][2] / rows[k][0])
            assert abs(
                Fraction(float(handle.weights[control_index])) - w_exact
            ) <= Fraction(1, 10**11) * max(Fraction(1), abs(w_exact))
            assert abs(
                Fraction(float(handle.control_points[control_index, 0]))
                - x_exact
            ) <= Fraction(1, 10**10) * max(Fraction(1), abs(x_exact))
            assert abs(
                Fraction(float(handle.control_points[control_index, 1]))
                - y_exact
            ) <= Fraction(1, 10**10) * max(Fraction(1), abs(y_exact))
        previous_rows = rows
