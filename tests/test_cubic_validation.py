"""Constructor input validation and scalar argument validation (spec 15, 19.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ph_spline import (
    ArcLengthOutOfRangeError,
    CubicPHSplineOpen,
    CubicPHSplineError,
    CubicPHSplineValueError,
    InsufficientPointDataError,
    InvalidPointDataError,
    NonFiniteCoordinateError,
    ParameterOutOfRangeError,
)

GOOD = [[0.0, 0.0], [1.0, 0.2], [2.0, 0.8], [2.5, 1.6]]


# ---------------------------------------------------------------------------
# Constructor container validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        None,
        42,
        "points",
        (np.array([[0.0, 0.0], [1.0, 1.0]])),
        ((0.0, 0.0), (1.0, 1.0)),  # outer tuple is rejected: must be a list
        {"a": 1},
    ],
)
def test_outer_container_must_be_list(bad):
    with pytest.raises(InvalidPointDataError):
        CubicPHSplineOpen(bad)


@pytest.mark.parametrize(
    "bad_element",
    [
        [0.0],
        [0.0, 1.0, 2.0],
        (0.0,),
        "xy",
        5.0,
        None,
        {"x": 0.0, "y": 1.0},
        np.array([0.0, 1.0]),  # ndarray element is not list/tuple
    ],
)
def test_malformed_point_element(bad_element):
    with pytest.raises(InvalidPointDataError):
        CubicPHSplineOpen([[0.0, 0.0], bad_element])


@pytest.mark.parametrize(
    "bad_coord",
    [True, False, np.bool_(True), "1.0", None, 1 + 2j, [1.0]],
)
def test_bad_coordinate_types(bad_coord):
    with pytest.raises(InvalidPointDataError):
        CubicPHSplineOpen([[0.0, 0.0], [bad_coord, 1.0]])


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, np.float64("nan")])
def test_nonfinite_coordinates(bad):
    with pytest.raises(NonFiniteCoordinateError):
        CubicPHSplineOpen([[0.0, 0.0], [1.0, bad]])


@pytest.mark.parametrize("pts", [[], [[0.0, 0.0]]])
def test_insufficient_points(pts):
    with pytest.raises(InsufficientPointDataError):
        CubicPHSplineOpen(pts)


def test_numpy_scalar_coordinates_accepted():
    pts = [[np.float64(0.0), np.int64(0)], (np.float32(1.0), np.float64(1.0))]
    curve = CubicPHSplineOpen(pts)
    assert curve.point(0.0).tolist() == [0.0, 0.0]


def test_open_constructor_exposes_documented_points_keyword():
    curve = CubicPHSplineOpen(points=GOOD)
    assert curve.num_points == len(GOOD)


def test_value_errors_are_value_errors():
    with pytest.raises(ValueError):
        CubicPHSplineOpen([[0.0, 0.0]])
    assert issubclass(InvalidPointDataError, CubicPHSplineError)
    assert issubclass(InvalidPointDataError, CubicPHSplineValueError)


# ---------------------------------------------------------------------------
# Scalar parameter validation (spec section 15.1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def curve():
    return CubicPHSplineOpen(GOOD)


ALL_U_METHODS = [
    "point",
    "tangent",
    "normal",
    "principal_normal",
    "signed_curvature",
    "curvature_vector",
    "arc_length",
]


@pytest.mark.parametrize("method", ALL_U_METHODS)
@pytest.mark.parametrize(
    "bad",
    [True, np.bool_(False), "0.5", None, 1 + 0j, [0.5], np.array([0.5]), np.array(0.5)],
)
def test_u_type_rejection(curve, method, bad):
    with pytest.raises(TypeError):
        getattr(curve, method)(bad)


@pytest.mark.parametrize("method", ALL_U_METHODS)
@pytest.mark.parametrize("bad", [math.nan, -0.1, 1.1, math.inf, -math.inf])
def test_u_domain_rejection(curve, method, bad):
    with pytest.raises(ParameterOutOfRangeError):
        getattr(curve, method)(bad)


@pytest.mark.parametrize("method", ALL_U_METHODS)
def test_u_accepts_python_and_numpy_scalars(curve, method):
    getattr(curve, method)(0)  # int
    getattr(curve, method)(0.25)
    getattr(curve, method)(np.float64(0.5))
    getattr(curve, method)(np.int32(1))


def test_u_ulp_clamping(curve):
    below = -2.0 * np.finfo(float).eps
    above = 1.0 + 2.0 * np.finfo(float).eps
    assert np.array_equal(curve.point(below), curve.point(0.0))
    assert np.array_equal(curve.point(above), curve.point(1.0))
    with pytest.raises(ParameterOutOfRangeError):
        curve.point(-1e-12)
    with pytest.raises(ParameterOutOfRangeError):
        curve.point(1.0 + 1e-12)


@pytest.mark.parametrize("method", ["parameter_at_length", "point_at_length"])
@pytest.mark.parametrize("bad", [True, "0.5", None, [0.5], np.array([0.5])])
def test_s_type_rejection(curve, method, bad):
    with pytest.raises(TypeError):
        getattr(curve, method)(bad)


@pytest.mark.parametrize("method", ["parameter_at_length", "point_at_length"])
def test_s_domain_rejection(curve, method):
    L = curve.arc_length(1.0)
    for bad in (math.nan, -0.1 * L, 1.1 * L, math.inf):
        with pytest.raises(ArcLengthOutOfRangeError):
            getattr(curve, method)(bad)


def test_s_ulp_clamping(curve):
    L = curve.arc_length(1.0)
    below = -math.ulp(L)
    above = L + 2.0 * math.ulp(L)
    assert curve.parameter_at_length(below) == 0.0
    assert curve.parameter_at_length(above) == 1.0
    with pytest.raises(ArcLengthOutOfRangeError):
        curve.parameter_at_length(L * (1.0 + 1e-9))


# ---------------------------------------------------------------------------
# Side argument (spec section 15.2)
# ---------------------------------------------------------------------------


def test_normal_side_values(curve):
    left = curve.normal(0.3, side="left")
    right = curve.normal(0.3, side="right")
    assert np.allclose(left, -right, rtol=0, atol=0)
    assert np.array_equal(curve.normal(0.3), left)  # default is left


@pytest.mark.parametrize("bad", ["Left", "RIGHT", "l", "", None, 1, b"left"])
def test_normal_side_rejection(curve, bad):
    with pytest.raises(ValueError):
        curve.normal(0.3, side=bad)


# ---------------------------------------------------------------------------
# Diagnostics content (spec section 16)
# ---------------------------------------------------------------------------


def test_exception_diagnostic_fields():
    try:
        CubicPHSplineOpen([[0.0, 0.0], [1.0, math.inf]])
    except NonFiniteCoordinateError as exc:
        assert exc.index == 1
        assert exc.quantity == "coordinate"
        assert "index=1" in str(exc)
    else:  # pragma: no cover
        pytest.fail("expected NonFiniteCoordinateError")
