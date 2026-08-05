"""Cyclic cubic-PH construction, seam continuity, and hardening."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ph_spline import (
    CubicPHSplineClosed,
    DegeneratePointDataError,
    InsufficientPointDataError,
    InterpolationDomainError,
)


def radial_points(
    count: int, *, amplitude: float = 0.0, frequency: int = 1, clockwise: bool = False
) -> list[list[float]]:
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    if clockwise:
        angles = -angles
    radii = 1.0 + amplitude * np.cos(frequency * angles)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles))).tolist()


@pytest.mark.parametrize("count", [3, 4, 5, 8, 17, 64])
@pytest.mark.parametrize("clockwise", [False, True])
def test_convex_cycles_interpolate_and_are_g2_at_every_join(count, clockwise):
    points = np.asarray(radial_points(count, clockwise=clockwise))
    curve = CubicPHSplineClosed(points.tolist())
    assert curve.closed is True
    assert curve.num_points == count
    assert curve.degree == 3
    for index, expected in enumerate(points):
        assert np.array_equal(curve.point(index / count), expected)
        left = curve._segments[index - 1]
        right = curve._segments[index]
        assert np.allclose(left.tangent_local(1.0), right.tangent_local(0.0), atol=1e-12)
        assert math.isclose(
            left.curvature_local(1.0),
            right.curvature_local(0.0),
            rel_tol=1e-10,
            abs_tol=1e-12,
        )


def test_closed_seam_position_frame_curvature_and_distance_are_periodic():
    curve = CubicPHSplineClosed(radial_points(19, amplitude=0.08, frequency=2))
    assert np.array_equal(curve.point(0.0), curve.point(1.0))
    assert np.allclose(curve.tangent(0.0), curve.tangent(1.0), atol=1e-12)
    assert np.allclose(curve.normal(0.0), curve.normal(1.0), atol=1e-12)
    assert math.isclose(
        curve.signed_curvature(0.0),
        curve.signed_curvature(1.0),
        rel_tol=1e-10,
        abs_tol=1e-12,
    )
    assert curve.arc_length(1.0) == curve.length
    assert curve.parameter_at_length(curve.length) == 1.0
    assert np.array_equal(curve.point_at_length(0.0), curve.point_at_length(curve.length))


def test_nonconvex_cycle_reuses_auxiliary_inflection_subsegments():
    points = radial_points(40, amplitude=0.55, frequency=5)
    first = CubicPHSplineClosed(points)
    second = CubicPHSplineClosed(points)
    assert len(first.aux_inflection_points) == 10
    assert first._joint_kinds.count("inflection") == 10
    assert first._joint_kinds[0] == "g2"
    assert np.array_equal(first.point(0.0), first.point(1.0))
    assert np.allclose(first.tangent(0.0), first.tangent(1.0), atol=1e-12)
    assert math.isclose(
        first.signed_curvature(0.0),
        first.signed_curvature(1.0),
        rel_tol=1e-10,
        abs_tol=1e-12,
    )
    for index, expected in enumerate(points):
        assert np.array_equal(first.point(index / len(points)), expected)
    for left, right in zip(first._segments, second._segments):
        assert np.array_equal(left.ctrl, right.ctrl)


def test_closed_input_lists_each_seam_point_once():
    with pytest.raises(InsufficientPointDataError):
        CubicPHSplineClosed([[0.0, 0.0], [1.0, 0.0]])
    points = radial_points(8)
    with pytest.raises(DegeneratePointDataError):
        CubicPHSplineClosed(points + [points[0]])


def test_closed_constructor_exposes_documented_points_keyword():
    curve = CubicPHSplineClosed(points=radial_points(8))
    assert curve.num_points == 8


def test_seam_that_is_a_straight_curved_transition_is_rejected():
    points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.5, 1.0], [0.0, 1.0]]
    with pytest.raises(InterpolationDomainError, match="seam"):
        CubicPHSplineClosed(points)
