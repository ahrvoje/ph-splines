"""Planar cubic Pythagorean-hodograph interpolation for general point data.

Public namespace: :class:`CubicPHSpline` and the documented exception
hierarchy rooted at :class:`CubicPHSplineError`.
"""

from cubic_ph_spline.exceptions import (
    ArcLengthInversionError,
    ArcLengthOutOfRangeError,
    CubicPHSplineError,
    CubicPHSplineRuntimeError,
    CubicPHSplineValueError,
    DegeneratePointDataError,
    G2VerificationError,
    InsufficientPointDataError,
    InterpolationDomainError,
    InvalidPointDataError,
    LengthResolutionError,
    NonAdmissibleSegmentError,
    NonFiniteCoordinateError,
    NonRegularSplineError,
    NonSimplePointDataError,
    NumericalPrecisionError,
    ParameterOutOfRangeError,
    ReversalError,
    SplineConvergenceError,
    UndefinedPrincipalNormalError,
)
from cubic_ph_spline.spline import CubicPHSpline

__version__ = "1.1.0"

__all__ = [
    "ArcLengthInversionError",
    "ArcLengthOutOfRangeError",
    "CubicPHSpline",
    "CubicPHSplineError",
    "CubicPHSplineRuntimeError",
    "CubicPHSplineValueError",
    "DegeneratePointDataError",
    "G2VerificationError",
    "InsufficientPointDataError",
    "InterpolationDomainError",
    "InvalidPointDataError",
    "LengthResolutionError",
    "NonAdmissibleSegmentError",
    "NonFiniteCoordinateError",
    "NonRegularSplineError",
    "NonSimplePointDataError",
    "NumericalPrecisionError",
    "ParameterOutOfRangeError",
    "ReversalError",
    "SplineConvergenceError",
    "UndefinedPrincipalNormalError",
]
