"""Analytic PHBSplineOpen length, inversion, relative travel, and batches."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ph_spline import LengthCoordinate, PHBSplineClosed, PHBSplineOpen

POINTS = [[0.0, 0.0], [1.0, 0.4], [2.0, -0.7], [3.0, 1.1], [2.2, 2.0]]


@pytest.fixture(scope="module", params=[2, 4, 8])
def curve(request):
    return PHBSplineOpen(POINTS, g_order=request.param)


def test_span_arc_coefficients_are_analytic_antiderivatives(curve):
    for span in curve._spans:
        degree = span.speed.size - 1
        assert span.arc.size == span.speed.size + 1
        assert span.arc[0] == 0.0
        assert span.arc[-1] == span.length
        assert np.allclose(
            np.diff(span.arc) * (degree + 1), span.speed, rtol=4e-14, atol=4e-14
        )


def test_global_length_endpoints_and_monotonicity(curve):
    assert curve.arc_length(0.0) == 0.0
    assert curve.arc_length(1.0) == curve.length
    values = [float(curve.arc_length(float(u))) for u in np.linspace(0.0, 1.0, 257)]
    assert np.all(np.diff(values) > 0.0)


def test_distance_round_trip_near_machine_precision(curve):
    rng = np.random.default_rng(20260805)
    targets = np.concatenate(
        (
            np.linspace(0.0, curve.length, 101),
            curve.length * rng.random(200),
            [math.ulp(curve.length), curve.length - math.ulp(curve.length)],
        )
    )
    tolerance = 512.0 * np.finfo(float).eps * curve.length
    for target in targets:
        u = curve.parameter_at_length(float(target))
        assert abs(float(curve.arc_length(u)) - target) <= tolerance + 8.0 * math.ulp(
            max(float(target), 1e-300)
        )


def test_point_at_length_matches_composition(curve):
    for distance in np.linspace(0.0, curve.length, 73):
        expected = curve.point(curve.parameter_at_length(float(distance)))
        assert np.allclose(
            curve.point_at_length(float(distance)),
            expected,
            rtol=2e-14,
            atol=2e-14,
        )


def test_extended_length_coordinate(curve):
    extended = curve.arc_length(0.37, extended=True)
    assert isinstance(extended, LengthCoordinate)
    assert float(extended) == curve.arc_length(0.37)
    assert curve.parameter_at_length(extended) == pytest.approx(0.37, abs=2e-13)


def test_distance_modes(curve):
    signed = curve.distance_between(0.8, 0.2, mode="signed")
    assert signed < 0.0
    assert curve.distance_between(0.8, 0.2) == -signed
    with pytest.raises(ValueError):
        curve.distance_between(0.2, 0.8, mode="bad")


def test_relative_location_travel_avoids_global_composition(curve):
    start = curve.location_at_length(0.4 * curve.length)
    delta = 1.0e-7 * curve.length
    advanced = curve.advance_by_length(start, delta)
    expected = curve.point_at_length(0.4 * curve.length + delta)
    assert np.allclose(curve.point(advanced), expected, rtol=2e-12, atol=2e-12)
    assert np.array_equal(curve.point_after_length(start, delta), curve.point(advanced))


def test_closed_forward_distance_wraps():
    angles = np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False)
    closed = PHBSplineClosed(np.column_stack((np.cos(angles), np.sin(angles))))
    expected = (
        closed.length + float(closed.arc_length(0.1)) - float(closed.arc_length(0.9))
    )
    assert closed.distance_between(0.9, 0.1, mode="forward") == pytest.approx(expected)


def test_batch_shapes_out_and_sorted_contract(curve):
    u = np.linspace(0.0, 1.0, 24).reshape(4, 6)
    result = curve.points_at(u)
    assert result.shape == (4, 6, 2)
    out = np.empty_like(result)
    assert curve.tangents_at(u, out=out) is out
    assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=2e-13)
    distances = np.linspace(0.0, curve.length, 24)
    assert curve.parameters_at_length(distances, assume_sorted=True).shape == (24,)
    assert curve.points_at_length(distances, assume_sorted=True).shape == (24, 2)
    with pytest.raises(ValueError):
        curve.points_at_length(distances[::-1], assume_sorted=True)
