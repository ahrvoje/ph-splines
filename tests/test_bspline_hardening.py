"""Adversarial PHBSpline construction and query hardening."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from ph_spline import PHBSpline, PHBSplineError


def test_geometric_spacing_uses_canonical_shared_knot_jets():
    lengths = [10.0 ** (-order) for order in range(12)]
    points = [[0.0, 0.0]]
    angle = 0.0
    for index, length in enumerate(lengths):
        if index:
            angle += 0.35
        points.append(
            [
                points[-1][0] + length * math.cos(angle),
                points[-1][1] + length * math.sin(angle),
            ]
        )
    curve = PHBSpline(points)
    assert curve.diagnostics.max_continuity_residual <= (
        curve.diagnostics.continuity_bound
    )
    assert np.all(np.isfinite(curve.points_at(np.linspace(0.0, 1.0, 65))))


@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize(
    "points",
    [
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [1.0, 1.0], [2.0, -1.0], [3.0, 1.0], [4.0, 0.0]],
        [[i, (-1.0) ** i] for i in range(20)],
        [[0.0, 0.0], [1.0, 0.0], [0.01, 0.01], [1.1, 0.02], [0.02, 0.03]],
        [[0.0, 0.0], [1.0, 1.0], [0.0, 2.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]],
        [[0.0, 0.0], [1.0e-10, 1.0], [1.0e4, 1.1], [1.0e4 + 1.0, -2.0]],
    ],
)
def test_named_adversarial_shapes_construct_without_warnings(points, order):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        curve = PHBSpline(points, g_order=order)
        assert curve.length > 0.0
        for u in np.linspace(0.0, 1.0, 33):
            assert np.all(np.isfinite(curve.point(float(u))))
            assert np.all(np.isfinite(curve.tangent(float(u))))


@pytest.mark.parametrize("seed", range(16))
def test_seeded_random_walk_fuzz(seed):
    rng = np.random.default_rng(seed)
    count = int(rng.integers(3, 40))
    directions = rng.uniform(-np.pi, np.pi, count - 1)
    lengths = 10.0 ** rng.uniform(-4.0, 4.0, count - 1)
    steps = np.column_stack((np.cos(directions), np.sin(directions))) * lengths[:, None]
    points = np.vstack((np.zeros(2), np.cumsum(steps, axis=0)))
    scale = 10.0 ** rng.uniform(-100.0, 100.0)
    points *= scale
    try:
        curve = PHBSpline(points, g_order=4)
    except PHBSplineError:
        # A typed regularity/precision failure is allowed for a path whose
        # chord dynamic range is deliberately adversarial.
        return
    targets = curve.length * rng.random(16)
    for target in targets:
        point = curve.point_at_length(float(target))
        assert np.all(np.isfinite(point))
        assert 0.0 <= curve.parameter_at_length(float(target)) <= 1.0


def test_long_curve_query_cost_does_not_change_results():
    x = np.linspace(0.0, 40.0, 1001)
    points = np.column_stack((x, np.sin(x)))
    curve = PHBSpline(points)
    targets = np.linspace(0.0, curve.length, 257)
    parameters = curve.parameters_at_length(targets, assume_sorted=True)
    assert np.all(np.diff(parameters) > 0.0)
    assert np.all(np.isfinite(curve.points_at_length(targets)))
