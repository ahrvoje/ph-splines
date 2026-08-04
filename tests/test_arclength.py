"""Arc-length and inversion tests (spec sections 12, 13, 14 and 19.4)."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from cubic_ph_spline import CubicPHSpline

EPS = np.finfo(float).eps


def segment_residual_bound(length: float, s: float) -> float:
    return 64.0 * EPS * length + 4.0 * math.ulp(s)


# ---------------------------------------------------------------------------
# Per-segment inversion (internal, exact spec bound)
# ---------------------------------------------------------------------------


def test_segment_inversion_exhaustive(any_case):
    curve = CubicPHSpline(any_case)
    rng = np.random.default_rng(20260804)
    for seg in curve._segments:
        L = seg.length
        # Endpoints must be exact.
        assert seg.invert_arc_length_local(0.0) == 0.0
        assert seg.invert_arc_length_local(L) == 1.0
        # Values within a few ulps of both endpoints.
        targets = [
            math.ulp(L),
            4.0 * math.ulp(L),
            L - math.ulp(L),
            L - 4.0 * math.ulp(L),
            L * 0.5,
            L * 0.25,
            L * 0.75,
        ]
        targets += list(L * rng.random(32))
        prev_t = -1.0
        for s in sorted(float(x) for x in targets):
            t = seg.invert_arc_length_local(s)
            assert 0.0 <= t <= 1.0
            assert math.isfinite(t)
            assert abs(seg.arc_length_local(t) - s) <= segment_residual_bound(L, s)
            # Monotonicity of the inverse.
            assert t >= prev_t
            prev_t = t


def test_segment_round_trip_t_to_s_to_t(any_case):
    curve = CubicPHSpline(any_case)
    for seg in curve._segments:
        for t in np.linspace(0.0, 1.0, 23):
            t = float(t)
            s = seg.arc_length_local(t)
            t_back = seg.invert_arc_length_local(s)
            # The inverse is well-conditioned: |dt| <= |ds| / sigma_min.
            s_min, _ = seg.sigma_extremes()
            tol = segment_residual_bound(seg.length, s) / s_min + 4.0 * EPS
            assert abs(t_back - t) <= tol


def test_inversion_from_both_traversal_directions(any_case):
    """Targets near both segment ends resolve with full relative accuracy."""
    curve = CubicPHSpline(any_case)
    for seg in curve._segments:
        L = seg.length
        for frac in (1e-15, 1e-12, 1e-9, 1e-3):
            s_near_start = L * frac
            t = seg.invert_arc_length_local(s_near_start)
            assert abs(seg.arc_length_local(t) - s_near_start) <= (
                segment_residual_bound(L, s_near_start)
            )
            s_near_end = L * (1.0 - frac)
            t = seg.invert_arc_length_local(s_near_end)
            assert abs(seg.arc_length_local(t) - s_near_end) <= (
                segment_residual_bound(L, s_near_end)
            )


# ---------------------------------------------------------------------------
# Global arc-length operations (public API)
# ---------------------------------------------------------------------------


def test_arc_length_endpoints(any_case):
    curve = CubicPHSpline(any_case)
    assert curve.arc_length(0.0) == 0.0
    L = curve.arc_length(1.0)
    assert L > 0.0
    assert curve.parameter_at_length(0.0) == 0.0
    assert curve.parameter_at_length(L) == 1.0
    assert np.array_equal(curve.point_at_length(0.0), curve.point(0.0))
    assert np.array_equal(curve.point_at_length(L), curve.point(1.0))


def test_arc_length_monotone(any_case):
    curve = CubicPHSpline(any_case)
    us = np.linspace(0.0, 1.0, 257)
    values = [curve.arc_length(float(u)) for u in us]
    assert all(b > a for a, b in pairwise(values))


def test_parameter_at_length_monotone(any_case):
    curve = CubicPHSpline(any_case)
    L = curve.arc_length(1.0)
    ss = np.linspace(0.0, L, 257)
    values = [curve.parameter_at_length(float(s)) for s in ss]
    assert all(b > a for a, b in pairwise(values))


def test_round_trip_s_u_s(any_case):
    curve = CubicPHSpline(any_case)
    L = curve.arc_length(1.0)
    rng = np.random.default_rng(7)
    targets = np.concatenate([np.linspace(0.0, L, 41), L * rng.random(64)])
    for s in targets:
        s = float(s)
        u = curve.parameter_at_length(s)
        assert 0.0 <= u <= 1.0
        s_back = curve.arc_length(u)
        # User-space slack: the spec residual plus normalization rounding.
        assert abs(s_back - s) <= 256.0 * EPS * L + 8.0 * math.ulp(max(s, 1e-300))


def test_prefix_lengths_map_to_exact_knots(any_case):
    curve = CubicPHSpline(any_case)
    m = curve._m
    for j in range(m + 1):
        s_j = curve.arc_length(j / m)
        assert curve.parameter_at_length(s_j) == j / m
        p = curve.point_at_length(s_j)
        assert np.array_equal(p, curve.point(j / m))


def test_point_at_length_matches_composition(any_case):
    curve = CubicPHSpline(any_case)
    L = curve.arc_length(1.0)
    for f in np.linspace(0.001, 0.999, 37):
        s = float(f) * L
        direct = curve.point_at_length(s)
        composed = curve.point(curve.parameter_at_length(s))
        assert np.allclose(direct, composed, rtol=1e-12, atol=1e-12 * L)


def test_point_at_length_is_c2_arclength_derivative(curved_case):
    """dr/ds == T within second-order finite-difference error."""
    curve = CubicPHSpline(curved_case)
    L = curve.arc_length(1.0)
    h = 1e-6 * L
    for f in (0.21, 0.4999, 0.5001, 0.83):
        s = f * L
        p_plus = curve.point_at_length(s + h)
        p_minus = curve.point_at_length(s - h)
        deriv = (p_plus - p_minus) / (2.0 * h)
        u = curve.parameter_at_length(s)
        T = curve.tangent(u)
        kappa = abs(curve.signed_curvature(u))
        # FD error ~ (h^2 / 6) |r'''| with |r'''| ~ max(kappa^2, |kappa'|).
        assert np.allclose(deriv, T, atol=1e-4 * max(1.0, kappa * kappa * L * L))


def test_arc_length_upper_bound_of_displacement(any_case):
    """||z(u2) - z(u1)|| <= arc length between them."""
    curve = CubicPHSpline(any_case)
    us = np.linspace(0.0, 1.0, 33)
    for u1, u2 in pairwise(us):
        chord = float(np.hypot(*(curve.point(float(u2)) - curve.point(float(u1)))))
        ds = curve.arc_length(float(u2)) - curve.arc_length(float(u1))
        assert chord <= ds * (1.0 + 1e-12) + 1e-300
