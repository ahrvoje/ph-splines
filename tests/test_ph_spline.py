"""Common PHSpline base-class contract."""

from __future__ import annotations

import numpy as np
import pytest

from ph_spline import CubicPHSpline, PHBSpline, PHSpline


@pytest.fixture(params=[CubicPHSpline, PHBSpline])
def curve(request) -> PHSpline:
    return request.param([[0.0, 0.0], [1.0, 0.4], [2.0, 1.1]])


def test_concrete_spline_classes_are_direct_common_base_subclasses():
    assert CubicPHSpline.__bases__ == (PHSpline,)
    assert PHBSpline.__bases__ == (PHSpline,)


def test_common_base_is_abstract():
    with pytest.raises(TypeError):
        PHSpline()


def test_polymorphic_geometry_and_distance_contract(curve: PHSpline):
    assert isinstance(curve, PHSpline)
    assert curve.degree >= 3
    assert curve.num_points == 3
    assert curve.closed is False
    assert curve.length == curve.arc_length(1.0)
    assert curve.point(0.4).shape == (2,)
    assert curve.tangent(0.4).shape == (2,)
    assert curve.normal(0.4).shape == (2,)
    assert np.isfinite(curve.signed_curvature(0.4))
    target = 0.37 * curve.length
    assert np.allclose(
        curve.point_at_length(target),
        curve.point(curve.parameter_at_length(target)),
        rtol=2e-13,
        atol=2e-13,
    )
