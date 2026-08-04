"""Invalid-data classification tests (spec sections 6 and 19.2)."""

from __future__ import annotations

import pytest
from conftest import circle_points, polyline_from_turns

from cubic_ph_spline import (
    CubicPHSpline,
    DegeneratePointDataError,
    InterpolationDomainError,
    NonSimplePointDataError,
    NumericalPrecisionError,
    ReversalError,
)


def test_consecutive_duplicates():
    with pytest.raises(DegeneratePointDataError) as exc_info:
        CubicPHSpline([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 3.0]])
    assert exc_info.value.index == (1, 2)


def test_signed_zero_duplicates_collide():
    with pytest.raises(DegeneratePointDataError):
        CubicPHSpline([[0.0, 0.0], [-0.0, -0.0]])


def test_nonconsecutive_duplicates():
    with pytest.raises(NonSimplePointDataError):
        CubicPHSpline([[0.0, 0.0], [1.0, 0.5], [2.0, 0.0], [0.0, 0.0]])


def test_collinear_backtracking():
    with pytest.raises(ReversalError):
        CubicPHSpline([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [2.0, 0.0]])


def test_mixed_straight_and_curved():
    # A straight two-span run followed by a two-span convex run.
    curve = CubicPHSpline([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 1.0], [3.0, 2.0]])
    assert curve._joint_kinds == ("g2", "transition", "g2")
    assert [seg.chi == 0.0 for seg in curve._segments] == [True, True, False, False]


def test_inflection_accepted():
    # Left turn followed by right turn.
    pts = polyline_from_turns([1.0, 1.0, 1.0], [0.5, -0.5])
    curve = CubicPHSpline(pts)
    assert len(curve._inflections) == 1
    assert len(curve._segments) == len(pts)


def test_near_pi_reversal():
    pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1e-15]]
    with pytest.raises(ReversalError):
        CubicPHSpline(pts)


def test_exact_reversal_on_line():
    with pytest.raises(ReversalError):
        CubicPHSpline([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]])


def test_self_intersecting_polyline():
    # A tight spiral whose chords cross: same-orientation turns but the
    # polyline passes through itself.
    pts = polyline_from_turns([1.0, 1.0, 1.0, 1.0, 1.0, 0.2], [1.8, 1.8, 1.8, 1.8, 3.0])
    with pytest.raises((NonSimplePointDataError, InterpolationDomainError)):
        CubicPHSpline(pts)


def test_self_intersection_direct():
    # Chords 0 and 2 properly cross while all turns keep one orientation
    # and the turn-pair bound is respected, so the crossing itself is the
    # only reason for rejection.
    pts = [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [-1.0, -3.0]]
    import numpy as np

    from cubic_ph_spline.predicates import polyline_self_intersection

    hit = polyline_self_intersection(np.array(pts))
    assert hit == (0, 2)
    with pytest.raises(NonSimplePointDataError):
        CubicPHSpline(pts)


def test_interior_uniqueness_violation():
    pts = polyline_from_turns([1.0, 1.0, 1.0], [2.06, 2.06])
    with pytest.raises(InterpolationDomainError) as exc_info:
        CubicPHSpline(pts)
    msg = str(exc_info.value)
    assert "4.0969" in msg  # required bound appears in diagnostics
    assert exc_info.value.value == pytest.approx(4.12)


def test_interior_uniqueness_just_below_bound_passes():
    pts = polyline_from_turns([1.0, 1.0, 1.0], [2.0, 2.0])
    CubicPHSpline(pts)


def test_chord_ratio_too_small():
    with pytest.raises(NumericalPrecisionError):
        CubicPHSpline([[0.0, 0.0], [1e-14, 1e-15], [1.0, 0.5], [1.5, 1.5]])


def test_overflow_during_normalization():
    with pytest.raises(NumericalPrecisionError):
        CubicPHSpline([[-1e308, 0.0], [1e308, 1e300], [1e308, 2e300]])


def test_clockwise_data_accepted():
    CubicPHSpline(circle_points(cw=True))


def test_counterclockwise_data_accepted():
    CubicPHSpline(circle_points(cw=False))
