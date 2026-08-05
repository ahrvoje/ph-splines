"""PHBSplineOpen input, policy, resource, and scalar validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ph_spline import (
    ArcLengthOutOfRangeError,
    ConstructionPolicy,
    ContinuitySpecificationError,
    DegeneratePointDataError,
    EditingPolicy,
    InsufficientPointDataError,
    InvalidPointDataError,
    InversePolicy,
    NonFiniteCoordinateError,
    NumericalPolicy,
    ParameterOutOfRangeError,
    PHBSplineClosed,
    PHBSplineOpen,
    PHBSplineError,
    ResourceLimitError,
)

POINTS = [[0.0, 0.0], [1.0, 0.4], [2.0, -0.7], [3.0, 1.1]]


@pytest.mark.parametrize(
    "container",
    [
        POINTS,
        tuple(tuple(row) for row in POINTS),
        np.asarray(POINTS),
        (row for row in POINTS),
    ],
)
def test_general_array_like_points_are_accepted(container):
    curve = PHBSplineOpen(container)
    assert curve.num_points == 4


@pytest.mark.parametrize("bad", [None, 3, "points", {"x": 1}, [], [[0.0, 0.0]]])
def test_malformed_or_insufficient_outer_data_rejected(bad):
    with pytest.raises((InvalidPointDataError, InsufficientPointDataError)):
        PHBSplineOpen(bad)


@pytest.mark.parametrize(
    "bad",
    [
        [[0.0], [1.0, 2.0]],
        [[0.0, 1.0, 2.0], [1.0, 2.0]],
        [[0.0, 0.0], [True, 1.0]],
        [[0.0, 0.0], ["1", 1.0]],
        [[0.0, 0.0], [1.0 + 2.0j, 1.0]],
    ],
)
def test_bad_point_elements_rejected(bad):
    with pytest.raises(InvalidPointDataError):
        PHBSplineOpen(bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_coordinates_rejected(bad):
    with pytest.raises(NonFiniteCoordinateError):
        PHBSplineOpen([[0.0, 0.0], [1.0, bad]])


def test_consecutive_duplicates_rejected_but_nonconsecutive_allowed():
    with pytest.raises(DegeneratePointDataError):
        PHBSplineOpen([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    curve = PHBSplineOpen([[0.0, 0.0], [1.0, 0.5], [0.0, 0.0], [-1.0, 0.5]])
    assert np.array_equal(curve.point(curve._knots[2]), [0.0, 0.0])


def test_closed_input_contract():
    with pytest.raises(InsufficientPointDataError):
        PHBSplineClosed([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(DegeneratePointDataError):
        PHBSplineClosed([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(TypeError):
        PHBSplineOpen(POINTS, closed=1)


@pytest.mark.parametrize("name", ["g_order", "c_order", "curvature_order"])
@pytest.mark.parametrize("bad", [True, -1, 1.5, "2", np.array(2)])
def test_continuity_order_validation(name, bad):
    with pytest.raises((ContinuitySpecificationError, TypeError)):
        PHBSplineOpen(POINTS, **{name: bad})


def test_degree_resource_limit_checked_before_construction():
    with pytest.raises(ResourceLimitError):
        PHBSplineOpen(POINTS, g_order=17, numerics=NumericalPolicy(max_preimage_degree=16))


@pytest.mark.parametrize(
    "policy",
    [
        NumericalPolicy(regularity_ratio_min=0.0),
        NumericalPolicy(max_preimage_degree=1),
        NumericalPolicy(max_evaluation_order=-1),
    ],
)
def test_invalid_numerical_policy_rejected(policy):
    with pytest.raises(ValueError):
        PHBSplineOpen(POINTS, numerics=policy)


def test_invalid_inverse_policy_rejected():
    with pytest.raises(ValueError):
        PHBSplineOpen(POINTS, inverse=InversePolicy(lut_nodes_min=16, lut_nodes_max=8))


@pytest.mark.parametrize(
    "keyword, policy",
    [
        ("construction", ConstructionPolicy(parameterization="bad")),
        ("construction", ConstructionPolicy(shape_objective="bad")),
        ("construction", ConstructionPolicy(initial_trust_radius=math.nan)),
        ("editing", EditingPolicy(initial_patch_spans=0)),
        ("editing", EditingPolicy(expansion_factor=math.inf)),
        ("inverse", InversePolicy(seed_kind="bad")),
        ("inverse", InversePolicy(endpoint_reverse_threshold=math.nan)),
        ("numerics", NumericalPolicy(position_eps_factor=math.nan)),
        ("numerics", NumericalPolicy(parameter_ulp_slack=-1)),
    ],
)
def test_every_policy_family_rejects_ill_defined_fields(keyword, policy):
    with pytest.raises(ValueError):
        PHBSplineOpen(POINTS, **{keyword: policy})


@pytest.fixture(scope="module")
def curve():
    return PHBSplineOpen(POINTS)


@pytest.mark.parametrize(
    "method",
    [
        "point",
        "tangent",
        "principal_normal",
        "signed_curvature",
        "curvature_vector",
        "arc_length",
        "derivative",
    ],
)
@pytest.mark.parametrize("bad", [True, "0.5", None, [0.5], np.array([0.5])])
def test_scalar_u_type_rejection(curve, method, bad):
    with pytest.raises(TypeError):
        getattr(curve, method)(bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.1, 1.1])
def test_scalar_u_domain_rejection(curve, bad):
    with pytest.raises(ParameterOutOfRangeError):
        curve.point(bad)


@pytest.mark.parametrize("bad", [True, "1", 1.5, -1, np.array(2)])
def test_derivative_order_rejection(curve, bad):
    with pytest.raises((TypeError, ValueError)):
        curve.derivative(0.3, bad)


def test_derivative_resource_limit(curve):
    with pytest.raises(ResourceLimitError):
        curve.derivative(0.3, 65)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0, 1.1])
def test_length_domain_rejection(curve, bad):
    with pytest.raises(ArcLengthOutOfRangeError):
        curve.point_at_length(bad * curve.length if math.isfinite(bad) else bad)


def test_points_property_is_a_read_only_copy(curve):
    first = curve.points
    second = curve.points
    assert first is not second
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[0, 0] = 10.0


def test_ph_exceptions_share_documented_root():
    assert issubclass(ResourceLimitError, PHBSplineError)
    assert issubclass(ParameterOutOfRangeError, PHBSplineError)
