"""Exception hierarchy for :mod:`cubic_ph_spline` (specification section 16).

Every construction exception carries structured diagnostic fields:

- ``index``   -- the relevant point or segment index,
- ``quantity`` -- the name of the failed quantity,
- ``value``   -- the measured value,
- ``bound``   -- the required bound.

The value branch derives from :class:`ValueError` and the runtime branch
from :class:`RuntimeError`, so idiomatic ``except ValueError`` /
``except RuntimeError`` handlers also work.
"""

from __future__ import annotations

__all__ = [
    "ArcLengthInversionError",
    "ArcLengthOutOfRangeError",
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


class CubicPHSplineError(Exception):
    """Base class of every exception raised by :mod:`cubic_ph_spline`."""

    def __init__(
        self,
        message: str,
        *,
        index: int | tuple[int, ...] | None = None,
        quantity: str | None = None,
        value: object = None,
        bound: object = None,
    ) -> None:
        self.index = index
        self.quantity = quantity
        self.value = value
        self.bound = bound
        details = []
        if index is not None:
            details.append(f"index={index}")
        if quantity is not None:
            details.append(f"quantity={quantity}")
        if value is not None:
            details.append(f"value={value!r}")
        if bound is not None:
            details.append(f"required bound={bound!r}")
        if details:
            message = f"{message} [{', '.join(details)}]"
        super().__init__(message)


class CubicPHSplineValueError(CubicPHSplineError, ValueError):
    """Invalid input data or invalid method arguments."""


class InvalidPointDataError(CubicPHSplineValueError):
    """The point container or a point element is malformed."""


class InsufficientPointDataError(CubicPHSplineValueError):
    """Fewer than two input points were supplied."""


class NonFiniteCoordinateError(CubicPHSplineValueError):
    """An input coordinate is NaN or infinite."""


class DegeneratePointDataError(CubicPHSplineValueError):
    """Consecutive input points coincide (a zero-length chord)."""


class NonSimplePointDataError(CubicPHSplineValueError):
    """A convex sub-polyline repeats a point or properly self-intersects."""


class ReversalError(CubicPHSplineValueError):
    """A turn angle of approximately pi (direction reversal or backtracking)."""


class InterpolationDomainError(CubicPHSplineValueError):
    """An interior turn-angle pair violates the uniqueness bound."""


class ParameterOutOfRangeError(CubicPHSplineValueError):
    """A global parameter ``u`` lies outside ``[0, 1]``."""


class ArcLengthOutOfRangeError(CubicPHSplineValueError):
    """An arc length ``s`` lies outside ``[0, L]``."""


class UndefinedPrincipalNormalError(CubicPHSplineValueError):
    """Principal normal requested on a completely straight spline."""


class CubicPHSplineRuntimeError(CubicPHSplineError, RuntimeError):
    """A verified numerical construction step could not be completed."""


class SplineConvergenceError(CubicPHSplineRuntimeError):
    """The nonlinear G2 system did not converge to an admissible solution."""


class NonAdmissibleSegmentError(CubicPHSplineRuntimeError):
    """A Bezier control polygon violates the orientation admissibility test."""


class NonRegularSplineError(CubicPHSplineRuntimeError):
    """A constructed PH segment is nearly cuspidal (speed too close to zero)."""


class G2VerificationError(CubicPHSplineRuntimeError):
    """Independent post-construction G2 or documented G1 verification failed."""


class ArcLengthInversionError(CubicPHSplineRuntimeError):
    """The safeguarded local arc-length inversion missed its residual bound."""


class LengthResolutionError(CubicPHSplineRuntimeError):
    """Prefix arc lengths are not strictly increasing in binary64."""


class NumericalPrecisionError(CubicPHSplineRuntimeError):
    """A quantity cannot be computed reliably in binary64 arithmetic."""
