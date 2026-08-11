"""Shared NURBS handle and offset-engine tests (spec 19.6 / 22.11).

These tests exercise the family-independent machinery: handle immutability
and validation, homogeneous de Boor evaluation against an independent
exact-rational evaluator, deterministic positive-weight refinement, and
forced refinement resource exhaustion.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from ph_spline import (
    CubicPHSplineOpen,
    NURBSHandle,
    OffsetConstructionError,
    ParameterOutOfRangeError,
)
from ph_spline.nurbs import (
    _MAX_REFINEMENT_DEPTH,
    _offset_patch,
    _refine_positive,
    _split_patch_half,
    build_offset_handle,
)

CURVE_POINTS = [[0.0, 0.0], [1.0, 0.4], [2.0, 1.3], [2.6, 2.4]]


@pytest.fixture(scope="module")
def handle():
    return CubicPHSplineOpen(CURVE_POINTS).offset(0.35)


# ---------------------------------------------------------------------------
# Construction protection and immutability
# ---------------------------------------------------------------------------


def test_direct_construction_is_forbidden(handle):
    with pytest.raises(TypeError, match="cannot be constructed directly"):
        NURBSHandle(
            degree=handle.degree,
            knots=handle.knots,
            control_points=handle.control_points,
            weights=handle.weights,
            num_spans=handle.num_spans,
            closed=False,
        )


def test_handle_attributes_are_immutable(handle):
    with pytest.raises(AttributeError):
        handle.degree = 7
    with pytest.raises(AttributeError):
        handle._homogeneous = None
    with pytest.raises(AttributeError):
        del handle._knots


def test_handle_arrays_are_readonly_snapshots(handle):
    for array in (handle.knots, handle.control_points, handle.weights):
        assert array.dtype == np.float64
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 0.0


def test_point_result_is_fresh_and_cannot_mutate_handle(handle):
    first = handle.point(0.4)
    first += 1000.0
    second = handle.point(0.4)
    assert abs(second[0] - (first[0] - 1000.0)) < 1e-12


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_point_rejects_non_scalar_arguments(handle):
    for bad in (True, np.True_, "0.5", [0.5], np.array([0.5, 0.6]), None):
        with pytest.raises(TypeError):
            handle.point(bad)


def test_point_rejects_nan_and_out_of_domain(handle):
    with pytest.raises(ParameterOutOfRangeError):
        handle.point(float("nan"))
    with pytest.raises(ParameterOutOfRangeError):
        handle.point(-0.001)
    with pytest.raises(ParameterOutOfRangeError):
        handle.point(1.001)


def test_point_clamps_few_ulp_violations(handle):
    below = -1.0e-16
    above = 1.0 + 2.0e-16
    assert np.array_equal(handle.point(below), handle.point(0.0))
    assert np.array_equal(handle.point(above), handle.point(1.0))


def test_point_accepts_numpy_scalars(handle):
    assert np.array_equal(handle.point(np.float64(0.25)), handle.point(0.25))


# ---------------------------------------------------------------------------
# Structural contract
# ---------------------------------------------------------------------------


def test_domain_and_shapes(handle):
    q = handle.degree
    n = handle.num_control_points
    assert handle.domain == (0.0, 1.0)
    assert handle.knots.shape == (n + q + 1,)
    assert handle.control_points.shape == (n, 2)
    assert handle.weights.shape == (n,)
    assert n == handle.num_spans * q + 1


def test_knot_multiplicities(handle):
    q = handle.degree
    knots = handle.knots
    assert np.all(knots[: q + 1] == 0.0)
    assert np.all(knots[-(q + 1) :] == 1.0)
    interior, counts = np.unique(knots[q + 1 : -(q + 1)], return_counts=True)
    assert np.all(counts == q)
    assert np.all(np.diff(knots) >= 0.0)
    assert interior.size == handle.num_spans - 1


def test_weights_strictly_positive_and_finite(handle):
    assert np.all(np.isfinite(handle.weights))
    assert np.all(handle.weights > 0.0)
    assert np.all(np.isfinite(handle.control_points))
    assert np.all(np.isfinite(handle.knots))


# ---------------------------------------------------------------------------
# Independent exact-rational evaluation of the published handle
# ---------------------------------------------------------------------------


def _exact_rational_point(handle: NURBSHandle, u: Fraction) -> tuple:
    """Evaluate the published NURBS with exact rational de Casteljau.

    This is completely independent of the production de Boor evaluator:
    it locates the Bezier span from the unique breakpoints and runs the
    de Casteljau recurrence in :class:`fractions.Fraction` arithmetic on
    the exact binary64 control data, so its only error is the final
    comparison rounding.
    """
    q = handle.degree
    breaks = [Fraction(b) for b in np.unique(handle.knots)]
    span = 0
    while span < len(breaks) - 2 and u >= breaks[span + 1]:
        span += 1
    width = breaks[span + 1] - breaks[span]
    t = (u - breaks[span]) / width
    controls = []
    for index in range(span * q, span * q + q + 1):
        w = Fraction(float(handle.weights[index]))
        controls.append(
            (
                w * Fraction(float(handle.control_points[index, 0])),
                w * Fraction(float(handle.control_points[index, 1])),
                w,
            )
        )
    s = 1 - t
    while len(controls) > 1:
        controls = [
            (
                s * a[0] + t * b[0],
                s * a[1] + t * b[1],
                s * a[2] + t * b[2],
            )
            for a, b in zip(controls[:-1], controls[1:])
        ]
    x, y, w = controls[0]
    return (x / w, y / w)


def test_de_boor_matches_exact_rational_evaluation(handle):
    rng = np.random.default_rng(20260811)
    parameters = [Fraction(1, 3), Fraction(1, 2), Fraction(7, 11)]
    parameters += [Fraction(float(v)) for v in rng.uniform(0.0, 1.0, 25)]
    for u in parameters:
        exact = _exact_rational_point(handle, u)
        produced = handle.point(float(u))
        scale = max(1.0, abs(float(exact[0])), abs(float(exact[1])))
        assert abs(float(exact[0]) - produced[0]) <= 1e-12 * scale
        assert abs(float(exact[1]) - produced[1]) <= 1e-12 * scale


# ---------------------------------------------------------------------------
# Positive-weight refinement mechanics
# ---------------------------------------------------------------------------


def _synthetic_patch() -> np.ndarray:
    """Homogeneous quintic patch with a positive denominator polynomial
    whose Bernstein weight coefficients are not all positive."""
    weights = np.array([1.0, -0.55, 1.0, 1.0, -0.55, 1.0])
    x = np.linspace(0.0, 1.0, 6) * weights
    y = np.zeros(6)
    return np.column_stack((weights, x, y))


def test_split_patch_half_partitions_exactly():
    patch = _synthetic_patch()
    left, right = _split_patch_half(patch)
    assert np.array_equal(left[0], patch[0])
    assert np.array_equal(right[-1], patch[-1])
    assert np.array_equal(left[-1], right[0])


def test_refinement_produces_positive_leaves():
    patch = _synthetic_patch()
    minimum = min(
        sum(
            w * math.comb(5, k) * t**k * (1.0 - t) ** (5 - k)
            for k, w in enumerate(patch[:, 0])
        )
        for t in np.linspace(0.0, 1.0, 2001)
    )
    assert minimum > 0.0  # the synthetic denominator really is positive
    leaves = _refine_positive(patch, 0.0, 1.0, 0, 0.1)
    assert len(leaves) > 1
    previous_hi = 0.0
    for lo, hi, data in leaves:
        assert lo == previous_hi
        assert hi > lo
        previous_hi = hi
        tau = np.max(np.abs(data[:, 0])) * 1e-10
        assert np.all(data[:, 0] > 0.0)
        assert np.all(data[:, 0] > -tau)
    assert previous_hi == 1.0


def test_refinement_is_deterministic():
    first = _refine_positive(_synthetic_patch(), 0.0, 1.0, 0, 0.1)
    second = _refine_positive(_synthetic_patch(), 0.0, 1.0, 0, -3.7)
    assert len(first) == len(second)
    for (lo_a, hi_a, data_a), (lo_b, hi_b, data_b) in zip(first, second):
        assert lo_a == lo_b and hi_a == hi_b
        assert np.array_equal(data_a, data_b)


def test_refinement_resource_exhaustion_raises():
    weights = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])  # crosses zero
    patch = np.column_stack((weights, weights, weights))
    with pytest.raises(OffsetConstructionError) as info:
        _refine_positive(patch, 0.0, 1.0, 3, 0.25)
    assert info.value.operation == "offset"
    assert info.value.refinement_depth == _MAX_REFINEMENT_DEPTH
    assert info.value.distance == 0.25


# ---------------------------------------------------------------------------
# Full synthetic pipeline through the refinement path
# ---------------------------------------------------------------------------


def _synthetic_rational_source():
    """One-span source whose declared speed has a negative Bernstein
    coefficient, exercising refinement inside the complete pipeline.

    The rational identity ``(sigma r + d R_L(v)) / sigma`` holds for any
    positive ``sigma``; the oracle below evaluates that identity directly
    with exact rational arithmetic, so the pipeline's verification gates
    remain meaningful for this synthetic pair.
    """
    controls = np.array(
        [[0.0, 0.0], [1.0, 0.6], [2.0, 0.6], [3.0, 0.0]], dtype=np.float64
    )
    # sigma(1/2) = 0.05 > 0, yet the degree-5 elevated Bernstein weight
    # W_2 = 0.4 + 0.6 * s_1 = -0.14 is negative, forcing refinement.
    speed = np.array([1.0, -0.9, 1.0])
    hodograph = 3.0 * np.diff(controls, axis=0)
    return controls, speed, hodograph


def _synthetic_oracle(controls, speed, distance):
    def _bernstein(values, t):
        work = [Fraction(float(v)) for v in values]
        s = 1 - t
        while len(work) > 1:
            work = [s * a + t * b for a, b in zip(work[:-1], work[1:])]
        return work[0]

    def oracle(u):
        t = Fraction(u)
        x = _bernstein(controls[:, 0], t)
        y = _bernstein(controls[:, 1], t)
        vx = _bernstein(3.0 * np.diff(controls[:, 0]), t)
        vy = _bernstein(3.0 * np.diff(controls[:, 1]), t)
        sigma = _bernstein(speed, t)
        d = Fraction(distance)
        return (
            (float(x + d * (-vy) / sigma), float(y + d * vx / sigma)),
            (0.0, 0.0),  # distance is folded into the point; normal unused
        )

    return oracle


def test_synthetic_pipeline_with_refinement():
    controls, speed, hodograph = _synthetic_rational_source()
    distance = 0.4
    oracle = _synthetic_oracle(controls, speed, distance)
    handle = build_offset_handle(
        span_controls=[controls],
        span_speeds=[speed],
        span_hodographs=[hodograph],
        hodograph_tolerance=1e-8,
        breakpoints=np.array([0.0, 1.0]),
        distance=distance,
        distance_normalized=distance,
        origin=(0.0, 0.0),
        scale=1.0,
        closed=False,
        join_tolerance=1e-12,
        oracle=lambda u: (oracle(u)[0], (0.0, 0.0)),
    )
    assert handle.num_spans > 1  # refinement really occurred
    assert np.all(handle.weights > 0.0)
    for u in (0.1, 0.25, 0.5, 0.61803, 0.9):
        expected = oracle(u)[0]
        produced = handle.point(u)
        assert math.hypot(
            produced[0] - expected[0], produced[1] - expected[1]
        ) <= 1e-11 * max(1.0, abs(expected[0]), abs(expected[1]))


def test_synthetic_pipeline_refinement_is_distance_independent():
    controls, speed, hodograph = _synthetic_rational_source()
    knots = []
    for distance in (-2.0, 0.0, 0.4):
        oracle = _synthetic_oracle(controls, speed, distance)
        handle = build_offset_handle(
            span_controls=[controls],
            span_speeds=[speed],
            span_hodographs=[hodograph],
            hodograph_tolerance=1e-8,
            breakpoints=np.array([0.0, 1.0]),
            distance=distance,
            distance_normalized=distance,
            origin=(0.0, 0.0),
            scale=1.0,
            closed=False,
            join_tolerance=1e-12,
            oracle=lambda u: (oracle(u)[0], (0.0, 0.0)),
        )
        knots.append((handle.knots, handle.weights))
    for other_knots, other_weights in knots[1:]:
        assert np.array_equal(knots[0][0], other_knots)
        assert np.array_equal(knots[0][1], other_weights)


def test_inconsistent_hodograph_is_rejected():
    controls, speed, hodograph = _synthetic_rational_source()
    corrupted = np.array(hodograph, copy=True)
    corrupted[1, 0] += 0.25
    oracle = _synthetic_oracle(controls, speed, 0.1)
    with pytest.raises(OffsetConstructionError) as info:
        build_offset_handle(
            span_controls=[controls],
            span_speeds=[speed],
            span_hodographs=[corrupted],
            hodograph_tolerance=1e-8,
            breakpoints=np.array([0.0, 1.0]),
            distance=0.1,
            distance_normalized=0.1,
            origin=(0.0, 0.0),
            scale=1.0,
            closed=False,
            join_tolerance=1e-12,
            oracle=lambda u: (oracle(u)[0], (0.0, 0.0)),
        )
    assert info.value.operation == "offset"


def test_offset_patch_matches_exact_rational_products():
    """Exact-arithmetic verification of the homogeneous coefficient
    formulas, independent of both production summation paths."""
    controls, speed, _ = _synthetic_rational_source()
    d_hat = -0.7
    patch = _offset_patch(controls, speed, d_hat, 0, d_hat)
    p = 3
    q = 5
    d = Fraction(d_hat)
    for k in range(q + 1):
        w_exact = Fraction(0)
        x_exact = Fraction(0)
        y_exact = Fraction(0)
        for i in range(max(0, k - p), min(p - 1, k) + 1):
            j = k - i
            lam = Fraction(math.comb(p - 1, i) * math.comb(p, j), math.comb(q, k))
            s_i = Fraction(float(speed[i]))
            vx = Fraction(3) * (
                Fraction(float(controls[i + 1, 0]))
                - Fraction(float(controls[i, 0]))
            )
            vy = Fraction(3) * (
                Fraction(float(controls[i + 1, 1]))
                - Fraction(float(controls[i, 1]))
            )
            w_exact += lam * s_i
            x_exact += lam * (s_i * Fraction(float(controls[j, 0])) + d * (-vy))
            y_exact += lam * (s_i * Fraction(float(controls[j, 1])) + d * vx)
        for produced, exact in (
            (patch[k, 0], w_exact),
            (patch[k, 1], x_exact),
            (patch[k, 2], y_exact),
        ):
            assert abs(Fraction(float(produced)) - exact) <= Fraction(1, 10**13)
