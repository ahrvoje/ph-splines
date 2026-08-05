"""Section-22 preprocessing, G1 joints, and general-data dispatch."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cubic_ph_spline import (
    CubicPHSpline,
    InterpolationDomainError,
    UndefinedPrincipalNormalError,
)

S_POINTS = [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]


def test_single_inflection_inserts_exactly_one_auxiliary_point():
    curve = CubicPHSpline(S_POINTS)
    assert len(curve._inflections) == 1
    assert len(curve._segments) == len(S_POINTS)
    info = curve._inflections[0]
    assert info.span_index == 1
    assert 1.0 / 16.0 <= info.rho <= 15.0 / 16.0
    u_aux = (info.span_index + info.rho) / (len(S_POINTS) - 1)
    j = int(np.flatnonzero(curve._knots == u_aux)[0])
    assert np.array_equal(curve.point(u_aux), np.array(info.point))
    assert np.array_equal(curve._segment_points[j], np.array(info.point))
    assert curve._joint_kinds[j - 1] == "inflection"


def test_aux_inflection_points_public_api():
    curve = CubicPHSpline(S_POINTS)
    points = curve.aux_inflection_points

    assert len(points) == 1
    assert set(points[0]) == {"u", "s", "x", "y"}
    u = points[0]["u"]
    s = points[0]["s"]
    assert np.array_equal(
        curve.point(u), np.array([points[0]["x"], points[0]["y"]])
    )
    assert s == curve.arc_length(u)
    assert curve.parameter_at_length(s) == u


def test_aux_inflection_points_is_empty_without_insertions():
    curve = CubicPHSpline([[0.0, 0.0], [1.0, 0.4], [2.0, 1.3], [2.6, 2.4]])
    assert curve.aux_inflection_points == []


def test_aux_inflection_points_returns_fresh_mutable_data():
    curve = CubicPHSpline(S_POINTS)
    points = curve.aux_inflection_points
    points[0]["x"] = math.inf
    points.append({"u": 0.0, "s": 0.0, "x": 0.0, "y": 0.0})

    assert len(curve.aux_inflection_points) == 1
    assert math.isfinite(curve.aux_inflection_points[0]["x"])


def test_inflection_joint_is_prescribed_g1_with_sign_flip():
    curve = CubicPHSpline(S_POINTS)
    j = curve._joint_kinds.index("inflection") + 1
    left, right = curve._segments[j - 1], curve._segments[j]
    d = np.array(curve._inflections[0].tangent)
    assert np.allclose(left.tangent_local(1.0), d, atol=1e-12, rtol=0.0)
    assert np.allclose(right.tangent_local(0.0), d, atol=1e-12, rtol=0.0)
    assert left.curvature_local(1.0) * right.curvature_local(0.0) < 0.0
    # Exact knots are right-sided for curvature and the principal normal.
    u_aux = float(curve._knots[j])
    assert curve.signed_curvature(u_aux) == pytest.approx(
        right.curvature_local(0.0) / curve._scale
    )
    expected = curve.normal(u_aux) * math.copysign(1.0, curve.signed_curvature(u_aux))
    assert np.allclose(curve.principal_normal(u_aux), expected, atol=1e-14)


def test_alternating_turns_insert_one_point_per_inflection_span():
    pts = [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]]
    curve = CubicPHSpline(pts)
    assert [info.span_index for info in curve._inflections] == [1, 2]
    assert len(curve._segments) == len(pts) - 1 + 2
    assert curve._joint_kinds.count("inflection") == 2
    assert [math.copysign(1.0, seg.chi) for seg in curve._segments] == [
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
    ]


def test_mixed_straight_curved_transition_is_g1_not_g2():
    pts = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 1.0], [3.0, 2.0]]
    curve = CubicPHSpline(pts)
    j = curve._joint_kinds.index("transition") + 1
    left, right = curve._segments[j - 1], curve._segments[j]
    assert left.chi == 0.0
    assert right.chi != 0.0
    assert np.allclose(left.tangent_local(1.0), right.tangent_local(0.0), atol=1e-12)
    assert left.curvature_local(1.0) == 0.0
    with pytest.raises(UndefinedPrincipalNormalError):
        curve.principal_normal(0.25)


def test_sign_change_separated_by_straight_run_has_no_auxiliary_point():
    pts = [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 2.0]]
    curve = CubicPHSpline(pts)
    assert curve._inflections == ()
    assert curve._joint_kinds == ("transition", "g2", "transition")
    assert [
        math.copysign(1.0, seg.chi) if seg.chi else 0.0 for seg in curve._segments
    ] == [1.0, 0.0, 0.0, -1.0]


def test_auxiliary_parameter_and_arc_length_round_trips():
    curve = CubicPHSpline(S_POINTS)
    for j, u in enumerate(curve._knots):
        s = curve.arc_length(float(u))
        assert curve.parameter_at_length(s) == float(u)
        assert np.array_equal(curve.point_at_length(s), curve._segment_points[j])
    for u in np.linspace(0.0, 1.0, 101):
        u = float(u)
        assert curve.parameter_at_length(curve.arc_length(u)) == pytest.approx(
            u, abs=2e-15
        )


def test_inflection_recipe_is_bitwise_deterministic():
    first = CubicPHSpline(S_POINTS)
    second = CubicPHSpline(S_POINTS)
    assert first._inflections == second._inflections
    assert np.array_equal(first._knots, second._knots)
    for a, b in zip(first._segments, second._segments):
        assert a.w0 == b.w0 and a.w1 == b.w1
        assert np.array_equal(a.ctrl, b.ctrl)


def test_deterministic_midpoint_tilt_fallback():
    pts = [[0.0, 0.0], [1.0, 0.0], [-2.0, -2.0], [0.0, -1.0]]
    curve = CubicPHSpline(pts)
    info = curve._inflections[0]
    assert info.fallback
    assert info.rho == 0.5
    d0 = np.array(pts[1]) - np.array(pts[0])
    d1 = np.array(pts[2]) - np.array(pts[1])
    d2 = np.array(pts[3]) - np.array(pts[2])
    phi_left = math.atan2(abs(d0[0] * d1[1] - d0[1] * d1[0]), float(d0 @ d1))
    phi_right = math.atan2(abs(d1[0] * d2[1] - d1[1] * d2[0]), float(d1 @ d2))
    assert info.delta == 0.5 * min(phi_left, phi_right, 0.5 * math.pi)
    chord = d1 / np.linalg.norm(d1)
    tangent = np.array(info.tangent)
    assert float(tangent @ chord) > 0.0
    assert abs(float(chord[0] * tangent[1] - chord[1] * tangent[0])) > 0.0
    assert all(seg.sigma_extremes()[0] > 0.0 for seg in curve._segments)


def test_nonconsecutive_duplicate_in_different_blocks_is_accepted():
    pts = [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [1.0, -1.0],
        [0.0, 0.0],
        [-1.0, -1.0],
        [-2.0, 0.0],
    ]
    curve = CubicPHSpline(pts)
    assert np.array_equal(curve.point(0.0), curve.point(4.0 / 6.0))


def test_chords_in_different_convex_blocks_may_cross():
    pts = [[0.0, 0.0], [-2.0, -2.0], [-1.0, 1.0], [-2.0, 1.0], [-1.0, -2.0]]
    curve = CubicPHSpline(pts)
    assert len(curve._inflections) == 1
    assert "inflection" in curve._joint_kinds


def test_prescribed_transition_tangent_is_never_clamped():
    headings = [0.0, 0.0, 2.1, 4.2]
    pts = [[0.0, 0.0]]
    for heading in headings:
        pts.append(
            [
                pts[-1][0] + math.cos(heading),
                pts[-1][1] + math.sin(heading),
            ]
        )
    with pytest.raises(InterpolationDomainError, match="Prescribed start tangent"):
        CubicPHSpline(pts)


def test_free_boundary_can_clamp_next_to_prescribed_single_segment():
    pts = [[0.0, 0.0]]
    for heading in (0.0, 2.8, 2.8):
        pts.append(
            [
                pts[-1][0] + math.cos(heading),
                pts[-1][1] + math.sin(heading),
            ]
        )
    curve = CubicPHSpline(pts)
    assert curve._boundary_clamped == (True, False)
    assert curve._joint_kinds == ("transition", "g2")
