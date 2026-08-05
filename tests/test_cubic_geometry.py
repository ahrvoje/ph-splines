"""Exact and near-exact geometry tests (spec section 19.1)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from conftest import circle_points, parabola_points

from ph_spline import CubicPHSpline


def test_knot_interpolation_is_exact(any_case):
    pts = any_case
    curve = CubicPHSpline(pts)
    m = len(pts) - 1
    for i, p in enumerate(pts):
        got = curve.point(i / m)
        assert got[0] == float(p[0]) and got[1] == float(p[1])


def test_segment_count(any_case):
    curve = CubicPHSpline(any_case)
    assert len(curve._segments) == len(any_case) - 1


def test_circle_curvature():
    R = 2.0
    curve = CubicPHSpline(circle_points(R=R, a0=-0.6, a1=1.2, n=13))
    for u in np.linspace(0.0, 1.0, 101):
        kappa = curve.signed_curvature(float(u))
        assert kappa == pytest.approx(1.0 / R, rel=2e-3)


def test_dense_circle_curvature_tightens():
    # Mid-segment curvature error decays quadratically with the angular
    # step (~0.08 * dtheta^2); n=81 over 1 rad gives dtheta = 0.0125.
    R = 5.0
    curve = CubicPHSpline(circle_points(R=R, a0=0.0, a1=1.0, n=81))
    for u in np.linspace(0.0, 1.0, 101):
        assert curve.signed_curvature(float(u)) == pytest.approx(1.0 / R, rel=3e-5)


def test_circle_arc_length():
    R = 3.0
    a0, a1 = 0.2, 1.9
    curve = CubicPHSpline(circle_points(R=R, a0=a0, a1=a1, n=41))
    assert curve.arc_length(1.0) == pytest.approx(R * (a1 - a0), rel=1e-6)


def test_parabola_section():
    pts = parabola_points(0.2, 2.0, 9)
    curve = CubicPHSpline(pts)
    m = len(pts) - 1
    # Interior-knot curvature approximates the exact parabola curvature
    # k = 2 / (1 + 4x^2)^1.5; the free boundaries follow the endpoint
    # tangent policy instead of the generating curve and are excluded.
    for i in range(2, m - 1):
        x = pts[i][0]
        expected = 2.0 / (1.0 + 4.0 * x * x) ** 1.5
        assert curve.signed_curvature(i / m) == pytest.approx(expected, rel=0.2)
    # Curvature decreases monotonically along this convex section.
    knot_kappas = [curve.signed_curvature(i / m) for i in range(1, m)]
    assert all(k > 0.0 for k in knot_kappas)
    assert all(b < a for a, b in pairwise(knot_kappas))


def test_translation_invariance():
    base = circle_points(R=2.0, a0=-0.5, a1=1.3, n=9)
    curve0 = CubicPHSpline(base)
    shift = (1234.5, -987.25)
    curve1 = CubicPHSpline([[p[0] + shift[0], p[1] + shift[1]] for p in base])
    for u in np.linspace(0.0, 1.0, 21):
        u = float(u)
        assert np.allclose(curve1.point(u) - shift, curve0.point(u), rtol=0, atol=1e-11)
        assert np.allclose(curve1.tangent(u), curve0.tangent(u), atol=1e-12)
        assert curve1.signed_curvature(u) == pytest.approx(
            curve0.signed_curvature(u), rel=1e-10
        )
    assert curve1.arc_length(1.0) == pytest.approx(curve0.arc_length(1.0), rel=1e-12)


def test_uniform_scaling_invariance():
    base = circle_points(R=2.0, a0=-0.5, a1=1.3, n=9)
    k = 1e6
    curve0 = CubicPHSpline(base)
    curve1 = CubicPHSpline([[k * p[0], k * p[1]] for p in base])
    for u in np.linspace(0.0, 1.0, 21):
        u = float(u)
        assert np.allclose(curve1.point(u) / k, curve0.point(u), rtol=1e-12, atol=1e-12)
        assert np.allclose(curve1.tangent(u), curve0.tangent(u), atol=1e-12)
        assert curve1.signed_curvature(u) * k == pytest.approx(
            curve0.signed_curvature(u), rel=1e-10
        )
    assert curve1.arc_length(1.0) / k == pytest.approx(
        curve0.arc_length(1.0), rel=1e-12
    )


def test_cw_ccw_mirror_symmetry():
    base = circle_points(R=1.5, a0=0.1, a1=1.7, n=8)
    ccw = CubicPHSpline(base)
    cw = CubicPHSpline([[p[0], -p[1]] for p in base])
    for u in np.linspace(0.0, 1.0, 17):
        u = float(u)
        assert cw.signed_curvature(u) == pytest.approx(
            -ccw.signed_curvature(u), rel=1e-10
        )
        pc, pm = ccw.point(u), cw.point(u)
        assert pm[0] == pytest.approx(pc[0], abs=1e-12)
        assert pm[1] == pytest.approx(-pc[1], abs=1e-12)


def test_tiny_curvature():
    R = 1e8
    curve = CubicPHSpline(circle_points(R=R, a0=0.0, a1=1e-3, n=6))
    for u in np.linspace(0.0, 1.0, 11):
        assert curve.signed_curvature(float(u)) == pytest.approx(1.0 / R, rel=1e-4)


def test_determinism():
    pts = circle_points(R=2.0, a0=-0.6, a1=1.2, n=9)
    c1 = CubicPHSpline(pts)
    c2 = CubicPHSpline(pts)
    us = np.linspace(0.0, 1.0, 40)
    for u in us:
        u = float(u)
        assert np.array_equal(c1.point(u), c2.point(u))
        assert c1.signed_curvature(u) == c2.signed_curvature(u)
        assert c1.arc_length(u) == c2.arc_length(u)


def test_boundary_clamp_flag_recorded():
    pts = circle_points(R=2.0, a0=-0.6, a1=1.2, n=9)
    curve = CubicPHSpline(pts)
    assert isinstance(curve._boundary_clamped, tuple)
    assert len(curve._boundary_clamped) == 2
