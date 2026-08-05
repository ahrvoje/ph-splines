"""Common PHSpline base-class contract."""

from __future__ import annotations

import numpy as np
import pytest

from ph_spline import (
    CubicPHSpline,
    CubicPHSplineClosed,
    CubicPHSplineOpen,
    PHBSpline,
    PHBSplineClosed,
    PHBSplineOpen,
    PHSpline,
)


@pytest.fixture(params=[CubicPHSplineOpen, PHBSplineOpen])
def curve(request) -> PHSpline:
    return request.param([[0.0, 0.0], [1.0, 0.4], [2.0, 1.1]])


def test_family_bases_are_siblings_and_concrete_topologies_are_siblings():
    assert CubicPHSpline.__bases__ == (PHSpline,)
    assert PHBSpline.__bases__ == (PHSpline,)
    assert CubicPHSplineOpen.__bases__ == (CubicPHSpline,)
    assert CubicPHSplineClosed.__bases__ == (CubicPHSpline,)
    assert PHBSplineOpen.__bases__ == (PHBSpline,)
    assert PHBSplineClosed.__bases__ == (PHBSpline,)
    assert not issubclass(CubicPHSplineOpen, CubicPHSplineClosed)
    assert not issubclass(CubicPHSplineClosed, CubicPHSplineOpen)
    assert not issubclass(PHBSplineOpen, PHBSplineClosed)
    assert not issubclass(PHBSplineClosed, PHBSplineOpen)


def test_common_base_is_abstract():
    for abstract in (PHSpline, CubicPHSpline, PHBSpline):
        with pytest.raises(TypeError):
            abstract()


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
