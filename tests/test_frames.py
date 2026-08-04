"""Differential-frame tests (spec section 19.5)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import circle_points

from cubic_ph_spline import CubicPHSpline

EPS = np.finfo(float).eps


def test_frame_identities_random_points(any_case):
    curve = CubicPHSpline(any_case)
    rng = np.random.default_rng(31415)
    for u in rng.random(64):
        u = float(u)
        T = curve.tangent(u)
        N = curve.normal(u)
        K = curve.curvature_vector(u)
        kappa = curve.signed_curvature(u)
        assert abs(math.hypot(*T) - 1.0) < 64.0 * EPS
        assert abs(math.hypot(*N) - 1.0) < 64.0 * EPS
        assert abs(float(T @ N)) < 64.0 * EPS
        assert math.hypot(*K) == pytest.approx(abs(kappa), rel=1e-13, abs=1e-300)
        # K = kappa * N_left exactly by construction.
        assert np.allclose(K, kappa * N, rtol=1e-13, atol=1e-300)


def test_tangent_matches_chord_direction_secantly(curved_case):
    """The tangent agrees with a symmetric secant to first order."""
    curve = CubicPHSpline(curved_case)
    h = 1e-7
    for u in (0.31, 0.5, 0.77):
        p_plus = curve.point(u + h)
        p_minus = curve.point(u - h)
        secant = p_plus - p_minus
        secant /= np.hypot(*secant)
        T = curve.tangent(u)
        assert np.allclose(secant, T, atol=1e-5)


def test_right_turning_sign_conventions():
    pts = circle_points(R=2.0, a0=0.2, a1=1.6, n=9, cw=True)
    curve = CubicPHSpline(pts)
    for u in np.linspace(0.0, 1.0, 21):
        u = float(u)
        kappa = curve.signed_curvature(u)
        assert kappa < 0.0
        NP = curve.principal_normal(u)
        NL = curve.normal(u, side="left")
        assert np.allclose(NP, -NL, rtol=0, atol=0)


def test_left_turning_sign_conventions():
    pts = circle_points(R=2.0, a0=0.2, a1=1.6, n=9, cw=False)
    curve = CubicPHSpline(pts)
    for u in np.linspace(0.0, 1.0, 21):
        u = float(u)
        assert curve.signed_curvature(u) > 0.0
        assert np.allclose(
            curve.principal_normal(u), curve.normal(u, side="left"), atol=0
        )


def test_principal_normal_points_to_center():
    R = 2.0
    center = np.array([10.0, -3.0])
    for cw in (False, True):
        pts = circle_points(R=R, a0=0.1, a1=1.5, n=11, cw=cw, center=tuple(center))
        curve = CubicPHSpline(pts)
        for u in np.linspace(0.0, 1.0, 13):
            u = float(u)
            p = curve.point(u)
            to_center = center - p
            to_center /= np.hypot(*to_center)
            NP = curve.principal_normal(u)
            # The PH spline is only near the circle, so allow a small angle.
            assert float(NP @ to_center) > 0.999


def test_curvature_not_finite_differenced():
    """Curvature is exact: it matches the analytic PH formula, not a stencil."""
    curve = CubicPHSpline(circle_points(R=1.0, a0=0.0, a1=1.4, n=8))
    seg = curve._segments[2]
    t = 0.37
    sigma = seg.sigma(t)
    expected = 2.0 * seg.chi / (sigma * sigma) / curve._scale
    u = (2 + t) / curve._m
    assert curve.signed_curvature(u) == pytest.approx(expected, rel=1e-12)
