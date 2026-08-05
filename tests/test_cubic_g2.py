"""G2 continuity verification at every join (spec section 19.3).

Joins are evaluated from BOTH segment sides through the internal segment
objects, independently of the public right-sided knot convention.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from ph_spline import CubicPHSplineOpen


def test_g2_at_all_joins(any_case):
    curve = CubicPHSplineOpen(any_case)
    segments = curve._segments
    for left, right in pairwise(segments):
        tl = left.tangent_local(1.0)
        tr = right.tangent_local(0.0)
        assert math.hypot(tl[0] - tr[0], tl[1] - tr[1]) < 1e-12
        kl = left.curvature_local(1.0)
        kr = right.curvature_local(0.0)
        denom = max(abs(kl), abs(kr), np.finfo(float).eps)
        assert abs(kl - kr) / denom < 1e-10


def test_public_two_sided_limits(curved_case):
    """Approaching a knot from both sides through the public API agrees."""
    curve = CubicPHSplineOpen(curved_case)
    m = curve._m
    for j in range(1, m):
        u = j / m
        h = 1e-9
        k_left = curve.signed_curvature(u - h)
        k_right = curve.signed_curvature(u + h)
        k_at = curve.signed_curvature(u)
        scale = max(abs(k_at), 1e-300)
        # Continuity: one-sided values differ from the knot value only by
        # O(h) parametric drift, far above roundoff but tiny for small h.
        assert abs(k_left - k_at) / scale < 1e-3
        assert abs(k_right - k_at) / scale < 1e-3
        t_left = curve.tangent(u - h)
        t_right = curve.tangent(u + h)
        assert np.allclose(t_left, t_right, atol=1e-4)


def test_signed_curvature_single_sign(curved_case):
    curve = CubicPHSplineOpen(curved_case)
    kappas = [curve.signed_curvature(float(u)) for u in np.linspace(0, 1, 201)]
    signs = {math.copysign(1.0, k) for k in kappas}
    assert len(signs) == 1


def test_chi_sign_uniform(curved_case):
    curve = CubicPHSplineOpen(curved_case)
    chis = [seg.chi for seg in curve._segments]
    assert all(c != 0.0 for c in chis)
    assert len({math.copysign(1.0, c) for c in chis}) == 1
