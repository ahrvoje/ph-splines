"""PHBSplineOpen interpolation, PH identities, high-order jets, and invariance."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ph_spline import DiscontinuousDerivativeError, PHBSplineClosed, PHBSplineOpen

POINTS = np.array([[0.0, 0.0], [1.0, 0.4], [2.0, -0.7], [3.0, 1.1], [2.2, 2.0]])


@pytest.mark.parametrize("order", [2, 3, 4, 6, 8])
def test_degree_and_verified_continuity(order):
    curve = PHBSplineOpen(POINTS, g_order=order)
    assert curve.preimage_degree == order
    assert curve.degree == 2 * curve.preimage_degree + 1
    assert curve.verified_continuity.g_order >= order
    assert curve.verified_continuity.c_order >= order
    assert curve.verified_continuity.curvature_order >= order - 2


@pytest.mark.parametrize(
    ("arguments", "required"),
    [
        ({}, 2),
        ({"g_order": 6}, 6),
        ({"c_order": 5}, 5),
        ({"curvature_order": 4}, 6),
        ({"g_order": 3, "c_order": 5, "curvature_order": 1}, 5),
    ],
)
def test_minimum_preimage_degree_is_deduced_from_all_constraints(arguments, required):
    curve = PHBSplineOpen(POINTS, **arguments)
    assert curve.preimage_degree == required
    assert curve.degree == 2 * required + 1


@pytest.mark.parametrize("order", [2, 4, 8])
def test_exact_interpolation_at_all_knots(order):
    curve = PHBSplineOpen(POINTS, g_order=order)
    for knot, expected in zip(curve._knots, POINTS):
        assert np.array_equal(curve.point(float(knot)), expected)


def test_compiled_hodograph_is_preimage_square():
    curve = PHBSplineOpen(POINTS)
    for span in curve._spans:
        for local in np.linspace(0.0, 1.0, 9):
            local = float(local)
            dz_dlocal = span.position_derivative_local(local, 1)
            expected = span.parameter_width * span.w(local) ** 2
            assert dz_dlocal == pytest.approx(expected, rel=2e-13, abs=2e-13)


@pytest.mark.parametrize("u", [0.07, 0.23, 0.51, 0.79, 0.94])
def test_parameter_first_derivative_matches_centered_difference(u):
    curve = PHBSplineOpen(POINTS, c_order=4)
    h = 1.0e-6
    finite_difference = (curve.point(u + h) - curve.point(u - h)) / (2.0 * h)
    assert np.allclose(curve.derivative(u), finite_difference, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("u", [0.11, 0.37, 0.83])
def test_intrinsic_derivative_and_curvature_identities(u):
    curve = PHBSplineOpen(POINTS, c_order=4)
    assert np.allclose(
        curve.derivative(u, 1, wrt="arc_length"), curve.tangent(u), atol=2e-13
    )
    curvature_vector = curve.curvature_vector(u)
    assert np.allclose(
        curvature_vector,
        curve.signed_curvature(u) * curve.normal(u),
        rtol=2e-11,
        atol=2e-11,
    )
    assert np.array_equal(curvature_vector, curve.derivative(u, 2, wrt="arc_length"))
    for order in range(4):
        assert np.array_equal(
            curve.curvature_vector(u, order),
            curve.derivative(u, order + 2, wrt="arc_length"),
        )


def test_full_jets_match_single_order_methods():
    curve = PHBSplineOpen(POINTS, c_order=6)
    for wrt in ("parameter", "arc_length"):
        jet = curve.jet(0.37, 6, wrt=wrt)
        assert len(jet) == 7
        for order, value in enumerate(jet):
            assert np.array_equal(value, curve.derivative(0.37, order, wrt=wrt))
    vector_jet = curve.curvature_vector_jet(0.37, 4)
    for order, value in enumerate(vector_jet):
        assert np.array_equal(value, curve.curvature_vector(0.37, order))


def test_parameter_derivatives_above_curve_degree_are_exact_zero():
    curve = PHBSplineOpen(POINTS)
    assert np.array_equal(curve.derivative(0.4, curve.degree + 1), [0.0, 0.0])


def test_join_side_semantics_follow_verified_order():
    curve = PHBSplineOpen(POINTS, c_order=4)
    knot = float(curve._knots[2])
    for order in range(5):
        left = curve.derivative(knot, order, side="left")
        right = curve.derivative(knot, order, side="right")
        scale = max(1.0, float(np.linalg.norm(left)), float(np.linalg.norm(right)))
        assert (
            np.linalg.norm(left - right) <= curve.diagnostics.continuity_bound * scale
        )
        curve.derivative(knot, order, side="auto")
    with pytest.raises(DiscontinuousDerivativeError):
        curve.derivative(knot, curve.preimage_degree + 1, side="auto")


def test_closed_seam_is_position_and_frame_continuous():
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    curve = PHBSplineClosed(points, c_order=4)
    assert np.array_equal(curve.point(0.0), curve.point(1.0))
    assert np.allclose(curve.tangent(0.0), curve.tangent(1.0), atol=2e-13)
    assert np.allclose(
        curve.curvature_vector(0.0), curve.curvature_vector(1.0), atol=2e-10
    )


def test_closed_g8_radial_star_preserves_twelve_fold_symmetry():
    points = []
    for index in range(24):
        angle = math.pi * index / 12.0
        radius = 1.0 if index % 2 == 0 else 1.9
        points.append([radius * math.cos(angle), radius * math.sin(angle)])
    curve = PHBSplineClosed(points, g_order=8)
    angle = math.pi / 6.0
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    for parameter in np.linspace(0.0, 1.0, 97, endpoint=False):
        rotated = rotation @ curve.point(float(parameter))
        advanced = curve.point(float((parameter + 1.0 / 12.0) % 1.0))
        assert np.allclose(advanced, rotated, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("scale", [1.0e-150, 1.0e-50, 1.0e50, 1.0e150, 1.0e307])
def test_power_scale_invariance(scale):
    curve = PHBSplineOpen(POINTS)
    scaled = PHBSplineOpen((POINTS * scale).tolist())
    for u in (0.0, 0.13, 0.51, 0.87, 1.0):
        assert np.allclose(
            scaled.point(u) / scale, curve.point(u), rtol=3e-13, atol=3e-13
        )
        assert np.allclose(scaled.tangent(u), curve.tangent(u), atol=3e-13)
        assert scaled.signed_curvature(u) * scale == pytest.approx(
            curve.signed_curvature(u), rel=3e-11, abs=1e-12
        )
    assert scaled.length / scale == pytest.approx(curve.length, rel=3e-13)


def test_translation_rotation_and_reflection_invariance():
    curve = PHBSplineOpen(POINTS)
    angle = 0.73
    matrix = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    offset = np.array([1.0e6, -2.0e6])
    transformed_points = POINTS @ matrix.T + offset
    transformed = PHBSplineOpen(transformed_points)
    reflected = PHBSplineOpen(POINTS * [1.0, -1.0])
    for u in (0.09, 0.37, 0.78):
        assert np.allclose(
            transformed.point(u), curve.point(u) @ matrix.T + offset, atol=2e-9
        )
        assert np.allclose(
            transformed.tangent(u), curve.tangent(u) @ matrix.T, atol=2e-11
        )
        assert reflected.signed_curvature(u) == pytest.approx(
            -curve.signed_curvature(u), rel=3e-11
        )
