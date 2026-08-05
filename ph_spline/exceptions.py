"""Exception hierarchy for :mod:`ph_spline` (specification section 16).

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
    "ConstructionConvergenceError",
    "ContinuitySpecificationError",
    "ContinuityVerificationError",
    "CubicPHSplineError",
    "CubicPHSplineRuntimeError",
    "CubicPHSplineValueError",
    "DegeneratePointDataError",
    "DiscontinuousDerivativeError",
    "G2VerificationError",
    "InsufficientPointDataError",
    "InterpolationDomainError",
    "InterpolationVerificationError",
    "InvalidPointDataError",
    "LengthResolutionError",
    "LocalEditFailure",
    "NonAdmissibleSegmentError",
    "NonFiniteCoordinateError",
    "NonRegularSplineError",
    "NonSimplePointDataError",
    "NumericalPrecisionError",
    "PHBSplineError",
    "PHBSplineRuntimeError",
    "PHBSplineValueError",
    "ParameterOutOfRangeError",
    "ResourceLimitError",
    "ReversalError",
    "SplineConvergenceError",
    "StaleHandleError",
    "StaleLocationError",
    "TransactionError",
    "UndefinedPrincipalNormalError",
]


class PHBSplineError(Exception):
    """Base class of every PH B-spline exception."""

    def __init__(
        self,
        message: str,
        *,
        index: int | tuple[int, ...] | None = None,
        point_id: int | None = None,
        span_id: int | None = None,
        operation: str | None = None,
        patch: tuple[int, int] | None = None,
        quantity: str | None = None,
        value: object = None,
        bound: object = None,
        iteration: int | None = None,
    ) -> None:
        self.index = index
        self.point_id = point_id
        self.span_id = span_id
        self.operation = operation
        self.patch = patch
        self.quantity = quantity
        self.value = value
        self.bound = bound
        self.iteration = iteration
        details = []
        if index is not None:
            details.append(f"index={index}")
        if point_id is not None:
            details.append(f"point_id={point_id}")
        if span_id is not None:
            details.append(f"span_id={span_id}")
        if operation is not None:
            details.append(f"operation={operation!r}")
        if patch is not None:
            details.append(f"patch={patch!r}")
        if quantity is not None:
            details.append(f"quantity={quantity}")
        if value is not None:
            details.append(f"value={value!r}")
        if bound is not None:
            details.append(f"required bound={bound!r}")
        if iteration is not None:
            details.append(f"iteration={iteration}")
        if details:
            message = f"{message} [{', '.join(details)}]"
        super().__init__(message)


class CubicPHSplineError(PHBSplineError):
    """Base class of every cubic PH-spline exception."""


class PHBSplineValueError(PHBSplineError, ValueError):
    """Invalid PH B-spline input data or query arguments."""


class CubicPHSplineValueError(PHBSplineValueError, CubicPHSplineError):
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


class PHBSplineRuntimeError(PHBSplineError, RuntimeError):
    """A verified PH B-spline numerical operation could not be completed."""


class CubicPHSplineRuntimeError(PHBSplineRuntimeError, CubicPHSplineError):
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


class ContinuitySpecificationError(PHBSplineValueError):
    """A requested continuity order or combination is invalid."""


class DiscontinuousDerivativeError(PHBSplineValueError):
    """An automatic join query has no unique continuous derivative value."""


class StaleHandleError(PHBSplineValueError):
    """A point handle no longer identifies a live interpolation point."""


class StaleLocationError(PHBSplineValueError):
    """A curve location belongs to an obsolete spline version."""


class ConstructionConvergenceError(PHBSplineRuntimeError):
    """PH B-spline construction could not find a verified regular branch."""


class InterpolationVerificationError(PHBSplineRuntimeError):
    """Compiled PH spans failed independent interpolation verification."""


class ContinuityVerificationError(PHBSplineRuntimeError):
    """Compiled PH spans failed independent continuity verification."""


class LocalEditFailure(PHBSplineRuntimeError):
    """A requested local repair failed without committing any mutation."""


class ResourceLimitError(PHBSplineRuntimeError):
    """A configured degree, order, refinement, or patch limit was exceeded."""


class TransactionError(PHBSplineRuntimeError):
    """An edit transaction was used or committed incorrectly."""
