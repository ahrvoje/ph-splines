"""Degenerate straight splines (spec sections 6.2, 11.5, 19.1, 19.5)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cubic_ph_spline import CubicPHSpline, UndefinedPrincipalNormalError


def test_two_point_line():
    curve = CubicPHSpline([[0.0, 0.0], [3.0, 4.0]])
    assert np.array_equal(curve.point(0.0), [0.0, 0.0])
    assert np.array_equal(curve.point(1.0), [3.0, 4.0])
    assert np.allclose(curve.point(0.5), [1.5, 2.0], rtol=1e-15, atol=0)
    assert curve.arc_length(1.0) == pytest.approx(5.0, rel=1e-15)
    assert np.allclose(curve.tangent(0.7), [0.6, 0.8], rtol=1e-15, atol=1e-15)


def test_collinear_multi_point(straight_case):
    pts = straight_case
    curve = CubicPHSpline(pts)
    m = len(pts) - 1
    for i, p in enumerate(pts):
        assert np.array_equal(curve.point(i / m), p)
    total = sum(
        math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        for i in range(m)
    )
    assert curve.arc_length(1.0) == pytest.approx(total, rel=1e-14)


def test_straight_curvature_zero(straight_case):
    curve = CubicPHSpline(straight_case)
    for u in np.linspace(0.0, 1.0, 17):
        assert curve.signed_curvature(float(u)) == 0.0
        assert np.array_equal(curve.curvature_vector(float(u)), [0.0, 0.0])


def test_straight_principal_normal_raises(straight_case):
    curve = CubicPHSpline(straight_case)
    with pytest.raises(UndefinedPrincipalNormalError):
        curve.principal_normal(0.5)


def test_straight_frames(straight_case):
    curve = CubicPHSpline(straight_case)
    eps = np.finfo(float).eps
    for u in np.linspace(0.0, 1.0, 9):
        T = curve.tangent(float(u))
        N = curve.normal(float(u))
        assert abs(np.hypot(*T) - 1.0) < 64 * eps
        assert abs(np.hypot(*N) - 1.0) < 64 * eps
        assert abs(float(T @ N)) < 64 * eps


def test_straight_arc_length_parametrization():
    pts = [[1.0, 1.0], [2.0, 2.0], [5.0, 5.0], [9.0, 9.0]]
    curve = CubicPHSpline(pts)
    L = curve.arc_length(1.0)
    for f in np.linspace(0.0, 1.0, 33):
        s = float(f) * L
        p = curve.point_at_length(s)
        expected = np.array([1.0, 1.0]) + s / math.sqrt(2.0)
        assert np.allclose(p, expected, rtol=1e-13, atol=1e-13)


def test_straight_parameter_at_length_knots():
    pts = [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [7.0, 0.0]]
    curve = CubicPHSpline(pts)
    # Exact prefix lengths map to exact knots.
    assert curve.parameter_at_length(0.0) == 0.0
    assert curve.parameter_at_length(curve.arc_length(1 / 3)) == 1 / 3
    assert curve.parameter_at_length(curve.arc_length(2 / 3)) == 2 / 3
    assert curve.parameter_at_length(curve.arc_length(1.0)) == 1.0
