"""Tests for ``min_curvature_radii`` on both spline families.

The reported radii must be at least as tight as any sampled curvature
(upper-bound property), agree with a high-precision local refinement, and
connect exactly to the offset cusp condition ``1 - d * kappa = 0``.
"""

from __future__ import annotations

import copy
import math
import pickle

import numpy as np
import pytest

from ph_spline import (
    CubicPHSplineClosed,
    CubicPHSplineOpen,
    PHBSplineClosed,
    PHBSplineOpen,
)

S_POINTS = [
    [0.0, 0.0], [1.0, 0.8], [2.0, 1.0], [3.0, 0.2], [4.0, 0.0], [5.0, 0.8],
]
WAVY_CLOSED = [
    [
        (2.0 + 0.3 * math.cos(5.0 * t)) * math.cos(t),
        (2.0 + 0.3 * math.cos(5.0 * t)) * math.sin(t),
    ]
    for t in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
]
PHB_POINTS = [[0, 0], [1, 0.5], [2, -0.3], [3, 0.8], [4, 0.1], [5, 1.0]]


def _dense_extremes(curve, samples=200001):
    kk = np.array([
        curve.signed_curvature(float(u))
        for u in np.linspace(1e-9, 1.0 - 1e-9, samples)
    ])
    k_left = float(kk.max())
    k_right = float(-kk.min())
    return (
        1.0 / k_left if k_left > 0.0 else math.inf,
        1.0 / k_right if k_right > 0.0 else math.inf,
    )


def _refined_max_curvature(curve, sign, samples=20001, rounds=200):
    """Refine every sampled local maximum to machine precision, keep the best.

    Near-symmetric curves carry several peaks that differ only at the 1e-8
    level, so refining only the coarse argmax can converge to a marginally
    lower twin; every local maximum is polished instead.
    """
    uu = np.linspace(1e-9, 1.0 - 1e-9, samples)
    kk = np.array([sign * curve.signed_curvature(float(u)) for u in uu])
    # Refine every sample close to the top: near-equal micro-crests can sit
    # closer together than the sampling step, so peak detection is not
    # reliable at the 1e-8 level.
    top = kk.max()
    candidates = [
        int(i)
        for i in np.nonzero(kk >= top - 1e-5 * max(1.0, abs(top)))[0]
    ]
    best = -math.inf
    for i in candidates:
        lo = float(uu[max(i - 1, 0)])
        hi = float(uu[min(i + 1, samples - 1)])
        for _ in range(rounds):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            if sign * curve.signed_curvature(m1) < sign * curve.signed_curvature(
                m2
            ):
                lo = m1
            else:
                hi = m2
        best = max(best, sign * curve.signed_curvature(0.5 * (lo + hi)))
    # a closed seam or an open end can carry the exact maximum
    best = max(best, sign * curve.signed_curvature(0.0),
               sign * curve.signed_curvature(1.0))
    return best


# ---------------------------------------------------------------------------
# Cubic family
# ---------------------------------------------------------------------------


def test_cubic_straight_reports_infinite_radii():
    line = CubicPHSplineOpen([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
    assert line.min_curvature_radii == (math.inf, math.inf)


def test_cubic_one_sided_arc():
    arc = CubicPHSplineOpen(
        [[math.cos(a), math.sin(a)] for a in np.linspace(0.0, 1.2, 9)]
    )
    rho_left, rho_right = arc.min_curvature_radii
    assert rho_right == math.inf
    dense_left, _ = _dense_extremes(arc)
    assert rho_left <= dense_left * (1.0 + 1e-12)  # upper-bound property
    assert abs(rho_left - dense_left) <= 1e-8 * dense_left


def test_cubic_clockwise_arc_swaps_sides():
    arc = CubicPHSplineOpen(
        [[math.cos(-a), math.sin(-a)] for a in np.linspace(0.0, 1.2, 9)]
    )
    rho_left, rho_right = arc.min_curvature_radii
    assert rho_left == math.inf and math.isfinite(rho_right)


@pytest.mark.parametrize(
    "make",
    [
        lambda: CubicPHSplineOpen(S_POINTS),
        lambda: CubicPHSplineClosed(WAVY_CLOSED),
    ],
    ids=["open_s_curve", "closed_wavy"],
)
def test_cubic_matches_refined_oracle(make):
    curve = make()
    rho_left, rho_right = curve.min_curvature_radii
    for sign, rho in ((+1.0, rho_left), (-1.0, rho_right)):
        assert math.isfinite(rho)
        refined = _refined_max_curvature(curve, sign)
        assert abs(1.0 / rho - refined) <= 5e-13 * refined


def test_cubic_closed_convex_inward_only():
    loop = CubicPHSplineClosed([[1, 0], [0, 1], [-1, 0], [0, -1]])
    rho_left, rho_right = loop.min_curvature_radii
    assert math.isfinite(rho_left) and rho_right == math.inf


def test_cubic_cusp_condition_reached_at_rho():
    curve = CubicPHSplineOpen(S_POINTS)
    rho_left, rho_right = curve.min_curvature_radii
    for d, sign in ((rho_left, +1.0), (-rho_right, -1.0)):
        refined = _refined_max_curvature(curve, sign)
        assert abs(1.0 - abs(d) * refined) <= 1e-10
        # cusp-critical and beyond-critical offsets still construct
        curve.offset(d)
        curve.offset(1.25 * d)


def test_cubic_offsets_inside_range_are_cusp_free():
    curve = CubicPHSplineOpen(S_POINTS)
    rho_left, _ = curve.min_curvature_radii
    d = 0.95 * rho_left
    handle = curve.offset(d)
    # the offset tangent magnitude |1 - d*kappa| stays strictly positive
    worst = min(
        abs(1.0 - d * curve.signed_curvature(float(u)))
        for u in np.linspace(0.0, 1.0, 20001)
    )
    assert worst > 0.04
    assert handle.degree == 5


def test_cubic_radii_survive_pickle_and_deepcopy():
    curve = CubicPHSplineOpen(S_POINTS)
    expected = curve.min_curvature_radii
    assert pickle.loads(pickle.dumps(curve)).min_curvature_radii == expected
    assert copy.deepcopy(curve).min_curvature_radii == expected


def test_cubic_deterministic():
    a = CubicPHSplineOpen(S_POINTS).min_curvature_radii
    b = CubicPHSplineOpen(S_POINTS).min_curvature_radii
    assert a == b


# ---------------------------------------------------------------------------
# PH B-spline family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("g_order", [2, 4], ids=["g2", "g4"])
def test_phb_matches_refined_oracle(g_order):
    curve = PHBSplineOpen(PHB_POINTS, g_order=g_order)
    rho_left, rho_right = curve.min_curvature_radii
    for sign, rho in ((+1.0, rho_left), (-1.0, rho_right)):
        assert math.isfinite(rho)
        refined = _refined_max_curvature(curve, sign)
        assert abs(1.0 / rho - refined) <= 5e-12 * refined


def test_phb_upper_bound_property():
    curve = PHBSplineOpen(PHB_POINTS)
    rho_left, rho_right = curve.min_curvature_radii
    dense_left, dense_right = _dense_extremes(curve)
    assert rho_left <= dense_left * (1.0 + 1e-12)
    assert rho_right <= dense_right * (1.0 + 1e-12)


def test_phb_closed_families():
    convex = PHBSplineClosed(
        [[1.0, 0.0], [0.31, 0.95], [-0.81, 0.59], [-0.81, -0.59],
         [0.31, -0.95]]
    )
    rho_left, rho_right = convex.min_curvature_radii
    assert math.isfinite(rho_left) and rho_right == math.inf
    wavy = PHBSplineClosed(WAVY_CLOSED)
    rho_left, rho_right = wavy.min_curvature_radii
    assert math.isfinite(rho_left) and math.isfinite(rho_right)
    refined = _refined_max_curvature(wavy, +1.0)
    assert abs(1.0 / rho_left - refined) <= 5e-12 * refined


def test_phb_cusp_condition_reached_at_rho():
    curve = PHBSplineOpen(PHB_POINTS)
    rho_left, _ = curve.min_curvature_radii
    refined = _refined_max_curvature(curve, +1.0)
    assert abs(1.0 - rho_left * refined) <= 1e-10
    curve.offset(rho_left)
    curve.offset(1.2 * rho_left)


def test_phb_cache_and_edit_invalidation():
    curve = PHBSplineOpen(PHB_POINTS)
    before = curve.min_curvature_radii
    assert curve.min_curvature_radii == before  # cached path
    curve.move_point(2, [2.0, -0.6])
    after = curve.min_curvature_radii
    assert after != before
    fresh = PHBSplineOpen(
        [[0, 0], [1, 0.5], [2.0, -0.6], [3, 0.8], [4, 0.1], [5, 1.0]]
    )
    assert after == pytest.approx(fresh.min_curvature_radii, rel=1e-12)


def test_phb_snapshot_keeps_pre_edit_radii():
    curve = PHBSplineOpen(PHB_POINTS)
    snapshot = curve.snapshot()
    before = snapshot.min_curvature_radii
    curve.move_point(2, [2.1, -0.5])
    assert snapshot.min_curvature_radii == before
    assert curve.min_curvature_radii != before
