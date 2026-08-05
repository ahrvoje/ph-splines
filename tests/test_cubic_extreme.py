"""Extreme-but-valid data: scaling, spacing, angles near bounds (spec 19.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import circle_points, polyline_from_turns, spiral_points

from ph_spline import CubicPHSpline

EPS = np.finfo(float).eps


def check_basic_contract(curve: CubicPHSpline) -> None:
    for left, right in zip(curve._segments[:-1], curve._segments[1:]):
        tl = left.tangent_local(1.0)
        tr = right.tangent_local(0.0)
        assert math.hypot(tl[0] - tr[0], tl[1] - tr[1]) < 1e-12
        kl, kr = left.curvature_local(1.0), right.curvature_local(0.0)
        assert abs(kl - kr) / max(abs(kl), abs(kr), EPS) < 1e-10
    L = curve.arc_length(1.0)
    assert math.isfinite(L) and L > 0.0
    for f in (0.1, 0.5, 0.9):
        u = curve.parameter_at_length(f * L)
        assert abs(curve.arc_length(u) - f * L) <= 256.0 * EPS * L + 8.0 * math.ulp(
            f * L
        )


def test_huge_coordinates():
    k = 1e120
    pts = [[k * p[0], k * p[1]] for p in circle_points(R=2.0, a0=0.0, a1=1.5, n=8)]
    curve = CubicPHSpline(pts)
    check_basic_contract(curve)
    assert curve.signed_curvature(0.5) == pytest.approx(1.0 / (2.0 * k), rel=5e-3)


def test_tiny_coordinates():
    k = 1e-120
    pts = [[k * p[0], k * p[1]] for p in circle_points(R=2.0, a0=0.0, a1=1.5, n=8)]
    curve = CubicPHSpline(pts)
    check_basic_contract(curve)
    assert curve.signed_curvature(0.5) == pytest.approx(1.0 / (2.0 * k), rel=5e-3)


def test_far_offset():
    pts = circle_points(R=100.0, a0=0.0, a1=1.2, n=9, center=(1e9, -1e9))
    curve = CubicPHSpline(pts)
    check_basic_contract(curve)


def test_extreme_chord_ratio():
    """Chord lengths spanning nine decades stay within the contract.

    Note: the data is built by summing prescribed chords from the origin so
    the tiny turns remain representable; sampling an analytic curve at
    1e-9 spacing would collapse to exactly collinear points in binary64.
    """
    lengths = list(10.0 ** np.linspace(-9.0, 0.0, 10))
    turns = [0.25] * 9
    pts = polyline_from_turns(lengths, turns)
    curve = CubicPHSpline(pts)
    check_basic_contract(curve)
    kappas = [curve.signed_curvature(float(u)) for u in np.linspace(0, 1, 65)]
    assert all(k > 0.0 for k in kappas)


def test_angles_close_to_uniqueness_bound():
    pts = polyline_from_turns([1.0, 1.0, 1.0], [2.04, 2.04])  # sum 4.08 < 4.0969
    curve = CubicPHSpline(pts)
    check_basic_contract(curve)


def test_single_large_turn_near_pi():
    pts = polyline_from_turns([1.0, 1.0, 1.0], [3.0, 0.5])
    curve = CubicPHSpline(pts)
    check_basic_contract(curve)


def test_long_spiral_many_points():
    pts = spiral_points(a=0.5, b=0.08, t0=0.0, t1=6 * math.pi, n=200)
    curve = CubicPHSpline(pts)
    check_basic_contract(curve)
    assert len(curve._segments) == 199


def test_large_system_sparse_solver_path():
    pts = circle_points(R=1.0, a0=0.0, a1=2.4, n=400)
    curve = CubicPHSpline(pts)
    check_basic_contract(curve)
    for u in np.linspace(0.0, 1.0, 41):
        assert curve.signed_curvature(float(u)) == pytest.approx(1.0, rel=1e-5)


def test_clockwise_spiral():
    pts = [[p[0], -p[1]] for p in spiral_points(n=60)]
    curve = CubicPHSpline(pts)
    check_basic_contract(curve)
    assert curve.signed_curvature(0.5) < 0.0


def test_scaling_extremes_of_same_shape_agree():
    base = circle_points(R=1.0, a0=0.1, a1=1.3, n=7)
    reference = CubicPHSpline(base)
    for k in (1e-100, 1e-10, 1e10, 1e100):
        scaled = CubicPHSpline([[k * p[0], k * p[1]] for p in base])
        for u in (0.25, 0.5, 0.75):
            assert scaled.tangent(u) == pytest.approx(reference.tangent(u), rel=1e-9)
            assert scaled.signed_curvature(u) * k == pytest.approx(
                reference.signed_curvature(u), rel=1e-9
            )
        assert scaled.arc_length(1.0) / k == pytest.approx(
            reference.arc_length(1.0), rel=1e-11
        )
