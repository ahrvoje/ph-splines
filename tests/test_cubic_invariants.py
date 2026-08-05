"""Post-construction invariants (spec section 18) and immutability (2.1)."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from ph_spline import CubicPHSplineOpen


def test_ph_identity(any_case):
    """z'(t) = w(t)^2 for every segment (invariant 4)."""
    curve = CubicPHSplineOpen(any_case)
    for seg in curve._segments:
        c = seg.ctrl
        for t in np.linspace(0.0, 1.0, 9):
            t = float(t)
            u = 1.0 - t
            # Bezier derivative: 3 * sum of edge vectors in Bernstein basis.
            dx = 3.0 * (
                u * u * (c[1, 0] - c[0, 0])
                + 2.0 * u * t * (c[2, 0] - c[1, 0])
                + t * t * (c[3, 0] - c[2, 0])
            )
            dy = 3.0 * (
                u * u * (c[1, 1] - c[0, 1])
                + 2.0 * u * t * (c[2, 1] - c[1, 1])
                + t * t * (c[3, 1] - c[2, 1])
            )
            w = seg.w_at(t)
            w2 = w * w
            scale = max(abs(w2), 1e-300)
            # Control-net endpoint snapping perturbs the identity by at
            # most the verified reconstruction residual.
            assert math.hypot(dx - w2.real, dy - w2.imag) / scale < 1e-9


def test_regularity(any_case):
    """sigma(t) > 0 on every segment (invariant 3)."""
    curve = CubicPHSplineOpen(any_case)
    for seg in curve._segments:
        s_min, s_end_max = seg.sigma_extremes()
        assert s_min > 0.0
        assert s_min / s_end_max > 1e-12
        for t in np.linspace(0.0, 1.0, 17):
            assert seg.sigma(float(t)) > 0.0


def test_strictly_increasing_arc_length(any_case):
    curve = CubicPHSplineOpen(any_case)
    prefix = curve._prefix
    assert all(b > a for a, b in pairwise(prefix))


def test_control_polygon_admissibility(curved_case):
    curve = CubicPHSplineOpen(curved_case)
    tau = curve._tau
    for seg in curve._segments:
        c = seg.ctrl
        e = np.diff(c, axis=0)
        assert tau * (e[0, 0] * e[1, 1] - e[0, 1] * e[1, 0]) > 0.0
        assert tau * (e[1, 0] * e[2, 1] - e[1, 1] * e[2, 0]) > 0.0


# ---------------------------------------------------------------------------
# Immutability (spec sections 2.1 and 2.3)
# ---------------------------------------------------------------------------


def test_instance_attributes_frozen():
    curve = CubicPHSplineOpen([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(AttributeError):
        curve._scale = 2.0
    with pytest.raises(AttributeError):
        curve.new_attr = 1
    with pytest.raises(AttributeError):
        del curve._scale


def test_constructor_copies_input():
    pts = [[0.0, 0.0], [1.0, 0.2], [2.0, 0.8], [2.5, 1.6]]
    curve = CubicPHSplineOpen(pts)
    before = curve.point(0.5).copy()
    pts[1][0] = 99.0  # mutate the caller's list after construction
    pts[2] = [5.0, -5.0]
    assert np.array_equal(curve.point(0.5), before)
    assert np.array_equal(curve.point(1 / 3), [1.0, 0.2])


def test_returned_arrays_do_not_alias_state():
    curve = CubicPHSplineOpen([[0.0, 0.0], [1.0, 0.2], [2.0, 0.8], [2.5, 1.6]])
    p = curve.point(0.0)
    p_orig = p.copy()
    p += 1000.0  # returned arrays are fresh; mutating them must be harmless
    q = curve.point(0.0)
    assert np.array_equal(q, p_orig)
    t = curve.tangent(0.4)
    t *= -1.0
    assert not np.array_equal(curve.tangent(0.4), t)


def test_internal_arrays_read_only():
    curve = CubicPHSplineOpen([[0.0, 0.0], [1.0, 0.2], [2.0, 0.8], [2.5, 1.6]])
    with pytest.raises(ValueError):
        curve._points[0, 0] = 1.0
    with pytest.raises(ValueError):
        curve._prefix[0] = 1.0
    for seg in curve._segments:
        with pytest.raises(ValueError):
            seg.ctrl[0, 0] = 1.0


def test_return_types(curved_case):
    curve = CubicPHSplineOpen(curved_case)
    u = 0.4
    for arr in (
        curve.point(u),
        curve.tangent(u),
        curve.normal(u),
        curve.principal_normal(u),
        curve.curvature_vector(u),
        curve.point_at_length(0.3 * curve.arc_length(1.0)),
    ):
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert arr.shape == (2,)
    assert type(curve.signed_curvature(u)) is float
    assert type(curve.arc_length(u)) is float
    assert type(curve.parameter_at_length(1e-3)) is float
