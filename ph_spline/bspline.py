"""PH B-spline family API and verified query dispatch."""

from __future__ import annotations

import math
import numbers
from contextlib import AbstractContextManager
from typing import ClassVar, Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ph_spline.base import PHSpline
from ph_spline.bspline_construction import (
    PHBSplineBuildState,
    build_ph_bspline_local_state,
    build_ph_bspline_state,
)
from ph_spline.bspline_segment import (
    bernstein_abs_square,
    bernstein_square,
)
from ph_spline.bspline_types import (
    BuildDiagnostics,
    ConstructionPolicy,
    ContinuitySpec,
    CurveLocation,
    EditingPolicy,
    EditRepair,
    EditReport,
    Frame2D,
    InsertResult,
    InversePolicy,
    LengthCoordinate,
    NumericalPolicy,
    PointHandle,
)
from ph_spline.exceptions import (
    ArcLengthOutOfRangeError,
    DiscontinuousDerivativeError,
    InvalidPointDataError,
    LocalEditFailure,
    NonFiniteCoordinateError,
    NumericalPrecisionError,
    OffsetConstructionError,
    ParameterOutOfRangeError,
    PHBSplineValueError,
    ResourceLimitError,
    StaleHandleError,
    StaleLocationError,
    TransactionError,
    UndefinedPrincipalNormalError,
)
from ph_spline.area import (
    source_signed_area,
    span_contribution_ball,
    statistics as _area_statistics,
)
from ph_spline.fill_area import fill_area_source
from ph_spline.nurbs import (
    ClosedNURBSHandle,
    NURBSHandle,
    build_offset_handle,
    validate_offset_distance,
)

_EPS = np.finfo(np.float64).eps


def _validate_scalar(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not a Boolean")
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar, not {type(value).__name__}")
    return float(value)


def _validate_order(value: object, limit: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("order must be a nonnegative integer scalar")
    order = int(value)
    if order < 0:
        raise ValueError("order must be nonnegative")
    if order > limit:
        raise ResourceLimitError(
            "Geometry evaluation order exceeds NumericalPolicy",
            quantity="order",
            value=order,
            bound=f"<= {limit}",
        )
    return order


def _series_product(a: NDArray, b: NDArray, size: int | None = None) -> NDArray:
    if size is None:
        size = min(a.size, b.size)
    return np.convolve(a, b)[:size]


def _series_reciprocal(values: NDArray) -> NDArray:
    result = np.empty_like(values)
    result[0] = 1.0 / values[0]
    for n in range(1, values.size):
        result[n] = -sum(values[k] * result[n - k] for k in range(1, n + 1)) / values[0]
    return result


def _series_derivative(values: NDArray) -> NDArray:
    return np.arange(1, values.size, dtype=np.float64) * values[1:]


def _edit_point(value: object) -> NDArray[np.float64]:
    if isinstance(value, (str, bytes, dict)) or value is None:
        raise InvalidPointDataError("Edit value must contain two real coordinates")
    try:
        coordinates = list(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise InvalidPointDataError(
            "Edit value must contain two real coordinates"
        ) from exc
    if len(coordinates) != 2:
        raise InvalidPointDataError(
            "Edit value must contain exactly two coordinates",
            quantity="coordinate count",
            value=len(coordinates),
            bound="== 2",
        )
    result = np.empty(2, dtype=np.float64)
    for axis, coordinate in enumerate(coordinates):
        if isinstance(coordinate, (bool, np.bool_)) or not isinstance(
            coordinate, numbers.Real
        ):
            raise InvalidPointDataError(
                "Edit coordinates must be real non-Boolean scalars",
                quantity=f"coordinate[{axis}]",
                value=coordinate,
            )
        converted = float(coordinate)
        if not math.isfinite(converted):
            raise NonFiniteCoordinateError(
                "Edit coordinates must be finite",
                quantity=f"coordinate[{axis}]",
                value=converted,
            )
        result[axis] = converted
    return result


class PHBSpline(PHSpline):
    """Abstract family base for mutable point-interpolating PH B-splines.

    The class constructs an immutable compiled PH span set before publishing
    any state.  Every edit builds and verifies a replacement first, then
    commits it atomically.  Its hot query path contains no numerical
    quadrature or geometric search.
    """

    def __init__(
        self,
        points: ArrayLike,
        *,
        _closed: bool,
        g_order: int | None = None,
        c_order: int | None = None,
        curvature_order: int | None = None,
        construction: ConstructionPolicy | None = None,
        editing: EditingPolicy | None = None,
        inverse: InversePolicy | None = None,
        numerics: NumericalPolicy | None = None,
    ) -> None:
        self._construction = construction or ConstructionPolicy()
        self._editing = editing or EditingPolicy()
        self._inverse = inverse or InversePolicy()
        self._numerics = numerics or NumericalPolicy()
        for name, policy, kind in (
            ("construction", self._construction, ConstructionPolicy),
            ("editing", self._editing, EditingPolicy),
            ("inverse", self._inverse, InversePolicy),
            ("numerics", self._numerics, NumericalPolicy),
        ):
            if not isinstance(policy, kind):
                raise TypeError(f"{name} must be {kind.__name__} or None")
        self._validate_policies()
        self._closed = _closed
        self._g_order = g_order
        self._c_order = c_order
        self._curvature_order = curvature_order
        state = self._build(points)
        self._version = 0
        self._handles = tuple(PointHandle(i) for i in range(state.points.shape[0]))
        self._next_handle_id = state.points.shape[0]
        self._last_edit_report: EditReport | None = None
        self._publish(state)

    def _validate_policies(self) -> None:
        construction = self._construction
        if construction.parameterization not in ("centripetal", "chord", "uniform"):
            raise ValueError("ConstructionPolicy.parameterization is invalid")
        construction_integers = (
            construction.max_iterations,
            construction.max_line_search_steps,
        )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in construction_integers
        ):
            raise ValueError(
                "ConstructionPolicy integer limits must be nonnegative integers"
            )

        if self._editing.default_repair not in ("strict_local", "expand", "global"):
            raise ValueError("EditingPolicy.default_repair is invalid")
        if (
            isinstance(self._editing.max_patch_spans, (bool, np.bool_))
            or not isinstance(self._editing.max_patch_spans, (int, np.integer))
            or self._editing.max_patch_spans < 1
            or (
                self._editing.initial_patch_spans is not None
                and (
                    isinstance(self._editing.initial_patch_spans, (bool, np.bool_))
                    or not isinstance(
                        self._editing.initial_patch_spans, (int, np.integer)
                    )
                    or self._editing.initial_patch_spans < 1
                )
            )
        ):
            raise ValueError("EditingPolicy patch limits must be positive")
        inverse_integers = (
            self._inverse.lut_nodes_min,
            self._inverse.lut_nodes_max,
            self._inverse.fast_iterations,
            self._inverse.max_iterations,
        )
        inverse_integer_types = all(
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, np.integer))
            for value in inverse_integers
        )
        threshold = self._inverse.endpoint_reverse_threshold
        if not (
            inverse_integer_types
            and 2 <= self._inverse.lut_nodes_min <= self._inverse.lut_nodes_max
            and self._inverse.max_iterations >= 1
            and self._inverse.fast_iterations >= 0
            and isinstance(threshold, numbers.Real)
            and not isinstance(threshold, (bool, np.bool_))
            and math.isfinite(float(threshold))
            and 0.0 <= threshold <= 1.0
            and isinstance(self._inverse.lut_power_of_two, (bool, np.bool_))
        ):
            raise ValueError("InversePolicy contains inconsistent limits")
        numerical_integers = (
            self._numerics.max_preimage_degree,
            self._numerics.max_evaluation_order,
            self._numerics.max_regularization_subdivision_depth,
            self._numerics.parameter_ulp_slack,
        )
        numerical_reals = (
            self._numerics.regularity_ratio_min,
            self._numerics.position_eps_factor,
            self._numerics.continuity_eps_factor,
        )
        if not (
            all(
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, (int, np.integer))
                for value in numerical_integers
            )
            and all(
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, numbers.Real)
                and math.isfinite(float(value))
                and float(value) > 0.0
                for value in numerical_reals
            )
            and 0.0 < self._numerics.regularity_ratio_min < 1.0
            and self._numerics.max_preimage_degree >= 2
            and self._numerics.max_evaluation_order >= 0
            and self._numerics.max_regularization_subdivision_depth >= 0
            and self._numerics.parameter_ulp_slack >= 0
            and self._numerics.position_eps_factor > 0.0
            and self._numerics.continuity_eps_factor > 0.0
            and isinstance(
                self._numerics.reject_unresolved_global_lengths, (bool, np.bool_)
            )
        ):
            raise ValueError("NumericalPolicy contains inconsistent limits")

    def _build(
        self,
        points: object,
        *,
        handles: tuple[PointHandle, ...] | None = None,
        reuse: bool = False,
    ) -> PHBSplineBuildState:
        point_ids = None
        if handles is not None:
            point_ids = tuple(handle.id for handle in handles)
        if reuse:
            assert point_ids is not None
            return build_ph_bspline_local_state(
                self._state,
                points,
                closed=self._closed,
                g_order=self._g_order,
                c_order=self._c_order,
                curvature_order=self._curvature_order,
                construction=self._construction,
                inverse=self._inverse,
                numerics=self._numerics,
                point_ids=point_ids,
            )
        return build_ph_bspline_state(
            points,
            closed=self._closed,
            g_order=self._g_order,
            c_order=self._c_order,
            curvature_order=self._curvature_order,
            construction=self._construction,
            inverse=self._inverse,
            numerics=self._numerics,
            point_ids=point_ids,
            reusable=None,
            normalization_frame=None,
        )

    def _publish(self, state: PHBSplineBuildState) -> None:
        self._state = state
        self._points = state.points
        self._spans = state.spans
        self._knots = state.knots
        self._span_knots = state.span_knots
        self._prefix_normalized = state.prefix_normalized
        self._prefix_user = state.prefix_user
        self._origin = state.origin
        self._scale = state.scale
        self._parameter_total = state.parameter_total
        self._total_length = state.total_length

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.num_points} points, {self.num_spans} spans, "
            f"degree={self.degree}, {'closed' if self.closed else 'open'}, "
            f"version={self.version})"
        )

    def __getstate__(self) -> dict:
        # Area caches are derived, nonauthoritative data and are omitted
        # from serialized state; restoration rebuilds them lazily on the
        # next query (area spec 11.4).
        state = dict(self.__dict__)
        state.pop("_area_cache", None)
        state.pop("_area_span_cache", None)
        state.pop("_fill_area_cache", None)
        return state

    # -- required properties ---------------------------------------------

    @property
    def points(self) -> NDArray[np.float64]:
        result = np.array(self._points, copy=True)
        result.setflags(write=False)
        return result

    @property
    def point_handles(self) -> tuple[PointHandle, ...]:
        return self._handles

    @property
    def num_points(self) -> int:
        return self._points.shape[0]

    @property
    def num_spans(self) -> int:
        return len(self._spans)

    @property
    def preimage_degree(self) -> int:
        return self._state.preimage_degree

    @property
    def degree(self) -> int:
        return 2 * self.preimage_degree + 1

    @property
    def requested_continuity(self) -> ContinuitySpec:
        return self._state.requested_continuity

    @property
    def verified_continuity(self) -> ContinuitySpec:
        return self._state.verified_continuity

    @property
    def length(self) -> float:
        return self._total_length

    @property
    def length_coordinate(self) -> LengthCoordinate:
        return LengthCoordinate(self._total_length, 0.0)

    @property
    def min_curvature_radii(self) -> tuple[float, float]:
        """Smallest left/right curvature radii ``(rho_left, rho_right)``.

        ``rho_left`` bounds the cusp-free positive offset range and
        ``rho_right`` the negative one: every ``offset(d)`` with
        ``-rho_right < d < rho_left`` is free of cusps, and equality reaches
        ``1 - d * kappa = 0`` exactly.  A side with no curvature of that
        sign reports ``math.inf``.  Per-span extremes are certified during
        span compilation (Bernstein sign-test subdivision of the critical
        polynomial); the merge over spans is cached per committed version,
        so repeated queries are O(1).
        """
        cached = self.__dict__.get("_min_radii_cache")
        if cached is not None and cached[0] == self._version:
            return cached[1]
        kappa_left = 0.0
        kappa_right = 0.0
        for span in self._spans:
            if span.kappa_max > kappa_left:
                kappa_left = span.kappa_max
            if -span.kappa_min > kappa_right:
                kappa_right = -span.kappa_min
        result = (
            self._scale / kappa_left if kappa_left > 0.0 else math.inf,
            self._scale / kappa_right if kappa_right > 0.0 else math.inf,
        )
        self.__dict__["_min_radii_cache"] = (self._version, result)
        return result

    @property
    def version(self) -> int:
        return self._version

    @property
    def diagnostics(self) -> BuildDiagnostics:
        return self._state.diagnostics

    @property
    def last_edit_report(self) -> EditReport | None:
        return self._last_edit_report

    # -- validation and dispatch -----------------------------------------

    def _validate_u(self, value: object) -> float:
        u = _validate_scalar("u", value)
        if math.isnan(u):
            raise ParameterOutOfRangeError("Parameter u must not be NaN", value=u)
        slack = self._numerics.parameter_ulp_slack * _EPS
        if u < 0.0:
            if u >= -slack:
                return 0.0
            raise ParameterOutOfRangeError(
                "Parameter u is below [0, 1]", quantity="u", value=u, bound=">= 0"
            )
        if u > 1.0:
            if u <= 1.0 + slack:
                return 1.0
            raise ParameterOutOfRangeError(
                "Parameter u is above [0, 1]", quantity="u", value=u, bound="<= 1"
            )
        return u

    def _validate_s(self, value: object) -> float:
        if isinstance(value, LengthCoordinate):
            s = float(value)
        else:
            s = _validate_scalar("s", value)
        if math.isnan(s):
            raise ArcLengthOutOfRangeError("Arc length must not be NaN", value=s)
        slack = self._numerics.parameter_ulp_slack * math.ulp(self._total_length)
        if s < 0.0:
            if s >= -slack:
                return 0.0
            raise ArcLengthOutOfRangeError(
                "Arc length is below [0, length]", value=s, bound=">= 0"
            )
        if s > self._total_length:
            if s <= self._total_length + slack:
                return self._total_length
            raise ArcLengthOutOfRangeError(
                "Arc length is above [0, length]",
                value=s,
                bound=f"<= {self._total_length!r}",
            )
        return s

    @staticmethod
    def _validate_join_side(side: object) -> Literal["auto", "left", "right"]:
        if side not in ("auto", "left", "right"):
            raise ValueError("side must be 'auto', 'left', or 'right'")
        return side  # type: ignore[return-value]

    def _join_index(self, u: float) -> int | None:
        if u == 0.0:
            return 0 if self._closed else None
        if u == 1.0:
            return 0 if self._closed else None
        index = int(np.searchsorted(self._span_knots, u, side="left"))
        if index < self._span_knots.size and u == self._span_knots[index]:
            return index
        return None

    def _locate(
        self, u: float, side: Literal["auto", "left", "right"] = "auto"
    ) -> tuple[int, float]:
        join = self._join_index(u)
        if join is not None:
            if self._closed and join == 0:
                if side == "left" or u == 1.0:
                    return len(self._spans) - 1, 1.0
                return 0, 0.0
            if side == "left":
                return join - 1, 1.0
            return join, 0.0
        if u == 0.0:
            if side == "left":
                raise ParameterOutOfRangeError("Open start has no left side")
            return 0, 0.0
        if u == 1.0:
            if side == "right":
                raise ParameterOutOfRangeError("Open end has no right side")
            return len(self._spans) - 1, 1.0
        span = int(np.searchsorted(self._span_knots, u, side="right") - 1)
        width = self._span_knots[span + 1] - self._span_knots[span]
        local = (u - self._span_knots[span]) / width
        return span, min(1.0, max(0.0, float(local)))

    def _user_complex(self, normalized: complex) -> complex:
        x = self._origin[0] + self._scale * normalized.real
        y = self._origin[1] + self._scale * normalized.imag
        if not (math.isfinite(x) and math.isfinite(y)):
            raise NumericalPrecisionError(
                "Geometry restoration overflowed user coordinates",
                quantity="restored point",
                value=(x, y),
            )
        return complex(x, y)

    @staticmethod
    def _vector(value: complex) -> NDArray[np.float64]:
        result = np.array([value.real, value.imag], dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise NumericalPrecisionError(
                "Geometry query produced a nonfinite vector", value=result.tolist()
            )
        return result

    def _point_on_span(self, span: int, local: float) -> NDArray[np.float64]:
        return self._vector(
            self._user_complex(self._spans[span].point_normalized(local))
        )

    # -- geometry ---------------------------------------------------------

    @overload
    def point(self, u: numbers.Real) -> NDArray[np.float64]: ...

    @overload
    def point(self, u: CurveLocation) -> NDArray[np.float64]: ...

    def point(self, u: object) -> NDArray[np.float64]:
        if isinstance(u, CurveLocation):
            self._validate_location(u)
            return self._point_on_span(u.span_id, u.local_u)
        value = self._validate_u(u)
        if value == 0.0:
            return np.array(self._points[0], copy=True)
        if value == 1.0:
            return np.array(self._points[0 if self._closed else -1], copy=True)
        join = self._join_index(value)
        if join is not None and join % 2 == 0:
            point_index = join // 2
            if point_index < self.num_points:
                return np.array(self._points[point_index], copy=True)
        span, local = self._locate(value)
        return self._point_on_span(span, local)

    def _parameter_derivative_on_span(
        self, span: int, local: float, order: int
    ) -> complex:
        if order == 0:
            return self._user_complex(self._spans[span].point_normalized(local))
        kernel = self._spans[span]
        if order > kernel.degree:
            return 0.0j
        # z_u = H w**2, where H is the total unnormalised parameter
        # weight.  Higher derivatives follow from the finite Leibniz sum,
        # avoiding cancellation in high-order position-control differences.
        product_order = order - 1
        value = sum(
            math.comb(product_order, j)
            * kernel.w_derivative(local, j)
            * kernel.w_derivative(local, product_order - j)
            for j in range(product_order + 1)
        )
        factor = (
            self._scale
            * self._parameter_total
            * (self._parameter_total / kernel.parameter_width) ** product_order
        )
        result = factor * value
        if not (math.isfinite(result.real) and math.isfinite(result.imag)):
            raise NumericalPrecisionError(
                "Parameter derivative is not representable",
                span_id=span,
                quantity="derivative",
                value=result,
            )
        return result

    def _w_series(self, span: int, local: float, order: int) -> NDArray[np.complex128]:
        kernel = self._spans[span]
        result = np.empty(order + 1, dtype=np.complex128)
        ratio = self._parameter_total / kernel.parameter_width
        for j in range(order + 1):
            if j > kernel.preimage_degree:
                result[j] = 0.0j
            else:
                derivative = kernel.w_derivative(local, j)
                result[j] = (
                    0.0j
                    if derivative == 0.0j
                    else derivative * ratio**j / math.factorial(j)
                )
        return result

    def _position_series(
        self, span: int, local: float, order: int
    ) -> NDArray[np.complex128]:
        result = np.empty(order + 1, dtype=np.complex128)
        result[0] = self._user_complex(self._spans[span].point_normalized(local))
        for j in range(1, order + 1):
            result[j] = self._parameter_derivative_on_span(
                span, local, j
            ) / math.factorial(j)
        return result

    def _speed_series(self, span: int, local: float, order: int) -> NDArray[np.float64]:
        w = self._w_series(span, local, order)
        product = _series_product(w, np.conjugate(w), order + 1)
        result = self._scale * self._parameter_total * product.real
        if result[0] <= 0.0 or not np.all(np.isfinite(result)):
            raise NumericalPrecisionError(
                "Curvature-vector recurrence lost positive speed",
                span_id=span,
                quantity="speed jet",
                value=result.tolist(),
                bound="finite with positive constant term",
            )
        return result

    def _join_query(
        self,
        u: float,
        side: Literal["auto", "left", "right"],
        evaluator,
    ):
        join = self._join_index(u)
        if side != "auto" or join is None:
            span, local = self._locate(u, side)
            return evaluator(span, local)
        left_span, left_local = self._locate(u, "left")
        right_span, right_local = self._locate(u, "right")
        left = evaluator(left_span, left_local)
        right = evaluator(right_span, right_local)
        difference = abs(left - right)
        scale = max(1.0, abs(left), abs(right))
        tolerance = (
            self._numerics.continuity_eps_factor
            * _EPS
            * max(1, self.preimage_degree**2)
            * scale
        )
        tolerance = max(
            tolerance,
            self._state.diagnostics.continuity_bound * scale,
        )
        if difference > tolerance:
            raise DiscontinuousDerivativeError(
                "The requested join derivative has distinct one-sided values",
                index=join,
                quantity="one-sided derivative gap",
                value=difference,
                bound=f"<= {tolerance:.3e}",
            )
        return left if abs(left) <= abs(right) else right

    def derivative(
        self,
        u: numbers.Real,
        order: int = 1,
        *,
        side: Literal["auto", "left", "right"] = "auto",
    ) -> NDArray[np.float64]:
        value = self._validate_u(u)
        derivative_order = _validate_order(order, self._numerics.max_evaluation_order)
        selected_side = self._validate_join_side(side)
        if derivative_order == 0:
            return self.point(value)
        result = self._join_query(
            value,
            selected_side,
            lambda span, local: self._parameter_derivative_on_span(
                span, local, derivative_order
            ),
        )
        return self._vector(result)

    def jet(
        self,
        u: numbers.Real,
        order: int,
        *,
        side: Literal["auto", "left", "right"] = "auto",
    ) -> tuple[NDArray[np.float64], ...]:
        value = self._validate_u(u)
        maximum = _validate_order(order, self._numerics.max_evaluation_order)
        selected_side = self._validate_join_side(side)
        join = self._join_index(value)
        if selected_side == "auto" and join is not None:
            return tuple(
                self.derivative(value, j, side=selected_side)
                for j in range(maximum + 1)
            )
        span, local = self._locate(value, selected_side)
        return tuple(
            self._vector(self._parameter_derivative_on_span(span, local, j))
            for j in range(maximum + 1)
        )

    def tangent(self, u: numbers.Real) -> NDArray[np.float64]:
        value = self._validate_u(u)
        span, local = self._locate(value)
        tx, ty = self._spans[span].tangent(local)
        return np.array([tx, ty], dtype=np.float64)

    def normal(
        self, u: numbers.Real, side: Literal["left", "right"] = "left"
    ) -> NDArray[np.float64]:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        tangent = self.tangent(u)
        result = np.array([-tangent[1], tangent[0]])
        return result if side == "left" else -result

    def _curvature_series(
        self, span: int, local: float, order: int
    ) -> NDArray[np.float64]:
        w = self._w_series(span, local, order + 1)
        dw = _series_derivative(w)
        cross_series = _series_product(np.conjugate(w), dw, order + 1).imag
        norm_series = _series_product(w, np.conjugate(w), order + 1).real
        reciprocal = _series_reciprocal(norm_series)
        reciprocal2 = _series_product(reciprocal, reciprocal, order + 1)
        return (
            2.0
            / (self._parameter_total * self._scale)
            * _series_product(cross_series, reciprocal2, order + 1)
        )

    def signed_curvature(self, u: numbers.Real) -> float:
        value = self._validate_u(u)
        span, local = self._locate(value)
        result = self._spans[span].signed_curvature_normalized(local) / self._scale
        if not math.isfinite(result):
            raise NumericalPrecisionError(
                "Signed curvature is not finite", value=result
            )
        return result

    def curvature_derivative(
        self,
        u: numbers.Real,
        order: int = 1,
        *,
        side: Literal["auto", "left", "right"] = "auto",
    ) -> float:
        value = self._validate_u(u)
        derivative_order = _validate_order(order, self._numerics.max_evaluation_order)
        selected_side = self._validate_join_side(side)

        def evaluate(span: int, local: float) -> float:
            series = self._curvature_series(span, local, derivative_order)
            return float(
                math.factorial(derivative_order) * series[derivative_order]
            )

        return float(self._join_query(value, selected_side, evaluate))

    def _curvature_vector_parameter_on_span(
        self, span: int, local: float, order: int
    ) -> complex:
        required = order + 2
        field = self._position_series(span, local, required)
        speed = self._speed_series(span, local, required)
        for _ in range(2):
            derivative = _series_derivative(field)
            field = _series_product(
                derivative,
                _series_reciprocal(speed[: derivative.size]),
                derivative.size,
            )
            speed = speed[: field.size]
        return complex(math.factorial(order) * field[order])

    def curvature_vector(
        self,
        u: numbers.Real,
        order: int = 0,
        *,
        side: Literal["auto", "left", "right"] = "auto",
    ) -> NDArray[np.float64]:
        value = self._validate_u(u)
        derivative_order = _validate_order(order, self._numerics.max_evaluation_order)
        selected_side = self._validate_join_side(side)
        result = self._join_query(
            value,
            selected_side,
            lambda span, local: self._curvature_vector_parameter_on_span(
                span, local, derivative_order
            ),
        )
        return self._vector(result)

    def curvature_vector_jet(
        self,
        u: numbers.Real,
        order: int,
        *,
        side: Literal["auto", "left", "right"] = "auto",
    ) -> tuple[NDArray[np.float64], ...]:
        maximum = _validate_order(order, self._numerics.max_evaluation_order)
        value = self._validate_u(u)
        selected_side = self._validate_join_side(side)
        join = self._join_index(value)
        if selected_side == "auto" and join is not None:
            return tuple(
                self.curvature_vector(value, j, side=selected_side)
                for j in range(maximum + 1)
            )
        span, local = self._locate(value, selected_side)
        field = self._position_series(span, local, maximum + 2)
        speed = self._speed_series(span, local, maximum + 2)
        for _ in range(2):
            derivative = _series_derivative(field)
            field = _series_product(
                derivative,
                _series_reciprocal(speed[: derivative.size]),
                derivative.size,
            )
            speed = speed[: field.size]
        return tuple(
            self._vector(math.factorial(j) * field[j]) for j in range(maximum + 1)
        )

    def principal_normal(self, u: numbers.Real) -> NDArray[np.float64]:
        curvature = self.signed_curvature(u)
        tolerance = 256.0 * _EPS / self._scale
        if abs(curvature) <= tolerance:
            raise UndefinedPrincipalNormalError(
                "Principal normal is undefined at zero curvature",
                quantity="signed curvature",
                value=curvature,
                bound=f"abs(kappa) > {tolerance:.3e}",
            )
        normal = self.normal(u)
        return normal if curvature > 0.0 else -normal

    # -- distance domain --------------------------------------------------

    def arc_length(
        self, u: numbers.Real, *, extended: bool = False
    ) -> float | LengthCoordinate:
        value = self._validate_u(u)
        if value == 0.0:
            result = 0.0
        elif value == 1.0:
            result = self._total_length
        else:
            join = self._join_index(value)
            if join is not None:
                result = float(self._prefix_user[join])
            else:
                span, local = self._locate(value)
                if local <= 0.5:
                    result = float(
                        self._prefix_user[span]
                        + self._scale * self._spans[span].arc_at(local)
                    )
                else:
                    result = float(
                        self._prefix_user[span + 1]
                        - self._scale * self._spans[span].arc_remainder(local)
                    )
        return LengthCoordinate(result, 0.0) if extended else result

    def _location_from_s(self, s: float) -> CurveLocation:
        if s == 0.0:
            return CurveLocation(0, 0.0, self._version)
        if s == self._total_length:
            return CurveLocation(len(self._spans) - 1, 1.0, self._version)
        normalized = s / self._scale
        index = int(
            np.searchsorted(self._prefix_normalized, normalized, side="right") - 1
        )
        if normalized == self._prefix_normalized[index]:
            return CurveLocation(index, 0.0, self._version)
        local_target = normalized - self._prefix_normalized[index]
        local = self._spans[index].invert_arc_length(local_target, self._inverse)
        return CurveLocation(index, local, self._version)

    def location_at_length(self, s: numbers.Real | LengthCoordinate) -> CurveLocation:
        return self._location_from_s(self._validate_s(s))

    def parameter_at_length(self, s: numbers.Real | LengthCoordinate) -> float:
        location = self.location_at_length(s)
        return float(
            self._span_knots[location.span_id]
            + location.local_u
            * (
                self._span_knots[location.span_id + 1]
                - self._span_knots[location.span_id]
            )
        )

    def point_at_length(
        self, s: numbers.Real | LengthCoordinate
    ) -> NDArray[np.float64]:
        return self.point(self.location_at_length(s))

    def frame_at_length(self, s: numbers.Real | LengthCoordinate) -> Frame2D:
        u = self.parameter_at_length(s)
        point = self.point(u)
        tangent = self.tangent(u)
        normal = self.normal(u)
        for array in (point, tangent, normal):
            array.setflags(write=False)
        return Frame2D(point, tangent, normal, self.signed_curvature(u))

    def distance_between(
        self,
        u0: numbers.Real,
        u1: numbers.Real,
        *,
        mode: Literal["absolute", "signed", "forward"] = "absolute",
    ) -> float:
        first = float(self.arc_length(u0))
        second = float(self.arc_length(u1))
        difference = second - first
        if mode == "signed":
            return difference
        if mode == "absolute":
            return abs(difference)
        if mode != "forward":
            raise ValueError("mode must be 'absolute', 'signed', or 'forward'")
        if difference >= 0.0:
            return difference
        if self._closed:
            return difference + self._total_length
        raise ArcLengthOutOfRangeError("Forward travel on an open spline is backward")

    def _validate_location(self, location: CurveLocation) -> None:
        if location.version != self._version:
            raise StaleLocationError(
                "CurveLocation belongs to an obsolete spline version",
                value=location.version,
                bound=f"== {self._version}",
            )
        if not (0 <= location.span_id < len(self._spans)) or not (
            0.0 <= location.local_u <= 1.0
        ):
            raise StaleLocationError("CurveLocation contains invalid local coordinates")

    def advance_by_length(
        self, location: CurveLocation, ds: numbers.Real
    ) -> CurveLocation:
        self._validate_location(location)
        remaining = _validate_scalar("ds", ds) / self._scale
        if not math.isfinite(remaining):
            raise ArcLengthOutOfRangeError("Relative travel must be finite")
        span = location.span_id
        local_s = self._spans[span].arc_at(location.local_u)
        target = local_s + remaining
        while target > self._spans[span].length:
            target -= self._spans[span].length
            span += 1
            if span == len(self._spans):
                if not self._closed:
                    raise ArcLengthOutOfRangeError(
                        "Relative travel exceeds open spline end"
                    )
                span = 0
        while target < 0.0:
            span -= 1
            if span < 0:
                if not self._closed:
                    raise ArcLengthOutOfRangeError(
                        "Relative travel precedes open spline start"
                    )
                span = len(self._spans) - 1
            target += self._spans[span].length
        local = self._spans[span].invert_arc_length(target, self._inverse)
        return CurveLocation(span, local, self._version)

    def point_after_length(
        self, location: CurveLocation, ds: numbers.Real
    ) -> NDArray[np.float64]:
        return self.point(self.advance_by_length(location, ds))

    # -- exact parallel offset (spec section 15.6) ------------------------

    def offset(self, distance: numbers.Real) -> NURBSHandle:
        """Exact rational NURBS parallel offset at a signed distance.

        Positive ``distance`` offsets along the left unit normal, negative
        along the right.  One complete committed source version is captured
        atomically; the returned immutable
        :class:`~ph_spline.nurbs.NURBSHandle` has degree
        ``4 * preimage_degree + 1`` and remains unchanged after later
        source edits.  Construction uses only finite homogeneous Bernstein
        products of the compiled PH span data; sampling and fitting are
        never used.
        """
        d = validate_offset_distance(distance)
        state = self._state
        spans = state.spans
        span_count = len(spans)
        origin = state.origin
        scale = state.scale
        closed = self._closed
        span_knots = np.asarray(state.span_knots, dtype=np.float64)
        join_tolerance = max(
            1.0e-9, 64.0 * state.diagnostics.continuity_bound
        )

        if d != 0.0:
            # The pointwise normal and the connected parallel curve are
            # unique only when every internal join and any closed seam
            # share one traversal unit tangent (spec 15.6.1).
            joins = list(range(1, span_count))
            if closed:
                joins.append(0)
            for j in joins:
                tl = spans[j - 1].tangent(1.0)
                tr = spans[j].tangent(0.0)
                gap = math.hypot(tl[0] - tr[0], tl[1] - tr[1])
                if not gap <= join_tolerance:
                    raise DiscontinuousDerivativeError(
                        "A nonzero offset requires one common unit tangent "
                        "at every join",
                        index=j,
                        quantity="|T-(1) - T+(0)|",
                        value=gap,
                        bound=f"<= {join_tolerance:.3e}",
                    )

        span_controls: list[NDArray[np.float64]] = []
        span_speeds: list[NDArray[np.float64]] = []
        span_hodographs: list[NDArray[np.float64]] = []
        for kernel in spans:
            position = kernel.position
            span_controls.append(
                np.column_stack((position.real, position.imag)).astype(
                    np.float64, copy=False
                )
            )
            # Independent product recomputation from the authoritative
            # preimage (spec 15.6.7): speed controls against h*|w|^2.
            width = kernel.parameter_width
            speed_check = width * bernstein_abs_square(kernel.preimage)
            speed_gap = float(np.max(np.abs(speed_check - kernel.speed)))
            speed_limit = 1.0e-8 * max(1.0, float(np.max(speed_check)))
            if not speed_gap <= speed_limit:
                raise OffsetConstructionError(
                    "Stored span speed controls disagree with h*|w|^2",
                    operation="offset",
                    span_id=kernel.span_id,
                    quantity="max |speed - h*|w|^2|",
                    value=speed_gap,
                    bound=f"<= {speed_limit:.3e}",
                    distance=d,
                )
            span_speeds.append(np.asarray(kernel.speed, dtype=np.float64))
            hodograph = width * bernstein_square(kernel.preimage)
            span_hodographs.append(
                np.column_stack((hodograph.real, hodograph.imag)).astype(
                    np.float64, copy=False
                )
            )

        def oracle(u: float) -> tuple[tuple[float, float], tuple[float, float]]:
            if u <= 0.0:
                span, local = 0, 0.0
            elif u >= 1.0:
                span, local = span_count - 1, 1.0
            else:
                span = int(np.searchsorted(span_knots, u, side="right")) - 1
                span = min(max(span, 0), span_count - 1)
                width = span_knots[span + 1] - span_knots[span]
                local = min(
                    1.0, max(0.0, float((u - span_knots[span]) / width))
                )
            kernel = spans[span]
            z = kernel.point_normalized(local)
            tx, ty = kernel.tangent(local)
            return (
                (origin[0] + scale * z.real, origin[1] + scale * z.imag),
                (-ty, tx),
            )

        return build_offset_handle(
            span_controls=span_controls,
            span_speeds=span_speeds,
            span_hodographs=span_hodographs,
            hodograph_tolerance=1.0e-8,
            breakpoints=span_knots,
            distance=d,
            distance_normalized=d / scale,
            origin=origin,
            scale=scale,
            closed=closed,
            join_tolerance=join_tolerance,
            oracle=oracle,
            # Distance metric certificate: the compiled complex Bernstein
            # preimage and stored local parameter width of every span.
            metric_preimages=[
                [complex(z) for z in kernel.preimage] for kernel in spans
            ],
            metric_widths=[float(kernel.parameter_width) for kernel in spans],
        )

    # -- explicit batches -------------------------------------------------

    @staticmethod
    def _prepare_batch(values: ArrayLike, name: str) -> NDArray[np.float64]:
        array = np.asarray(values)
        if array.dtype == np.bool_ or array.ndim == 0:
            raise TypeError(f"{name} must be a non-scalar real array")
        try:
            return np.asarray(array, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a real array") from exc

    @staticmethod
    def _finish_batch(
        result: NDArray[np.float64], out: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        if out is None:
            return result
        if (
            not isinstance(out, np.ndarray)
            or out.shape != result.shape
            or out.dtype != np.float64
        ):
            raise ValueError(
                "out must be a writable float64 array of the exact result shape"
            )
        if not out.flags.writeable:
            raise ValueError("out must be writable")
        out[...] = result
        return out

    def points_at(
        self, u: ArrayLike, *, out: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        values = self._prepare_batch(u, "u")
        result = np.empty(values.shape + (2,), dtype=np.float64)
        for index in np.ndindex(values.shape):
            result[index] = self.point(float(values[index]))
        return self._finish_batch(result, out)

    def tangents_at(
        self, u: ArrayLike, *, out: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        values = self._prepare_batch(u, "u")
        result = np.empty(values.shape + (2,), dtype=np.float64)
        for index in np.ndindex(values.shape):
            result[index] = self.tangent(float(values[index]))
        return self._finish_batch(result, out)

    def derivatives_at(
        self,
        u: ArrayLike,
        order: int = 1,
        *,
        side: Literal["auto", "left", "right"] = "auto",
        out: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        values = self._prepare_batch(u, "u")
        result = np.empty(values.shape + (2,), dtype=np.float64)
        for index in np.ndindex(values.shape):
            result[index] = self.derivative(float(values[index]), order, side=side)
        return self._finish_batch(result, out)

    def curvature_vectors_at(
        self,
        u: ArrayLike,
        order: int = 0,
        *,
        side: Literal["auto", "left", "right"] = "auto",
        out: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        values = self._prepare_batch(u, "u")
        result = np.empty(values.shape + (2,), dtype=np.float64)
        for index in np.ndindex(values.shape):
            result[index] = self.curvature_vector(
                float(values[index]), order, side=side
            )
        return self._finish_batch(result, out)

    def points_at_length(
        self,
        s: ArrayLike,
        *,
        out: NDArray[np.float64] | None = None,
        assume_sorted: bool = False,
    ) -> NDArray[np.float64]:
        values = self._prepare_batch(s, "s")
        if not isinstance(assume_sorted, (bool, np.bool_)):
            raise TypeError("assume_sorted must be Boolean")
        result = np.empty(values.shape + (2,), dtype=np.float64)
        if assume_sorted:
            locations = self._sorted_length_locations(values)
            for index, location in zip(
                np.ndindex(values.shape), locations, strict=True
            ):
                result[index] = self._point_on_span(location.span_id, location.local_u)
            return self._finish_batch(result, out)
        for index in np.ndindex(values.shape):
            result[index] = self.point_at_length(float(values[index]))
        return self._finish_batch(result, out)

    def parameters_at_length(
        self,
        s: ArrayLike,
        *,
        out: NDArray[np.float64] | None = None,
        assume_sorted: bool = False,
    ) -> NDArray[np.float64]:
        values = self._prepare_batch(s, "s")
        if not isinstance(assume_sorted, (bool, np.bool_)):
            raise TypeError("assume_sorted must be Boolean")
        result = np.empty(values.shape, dtype=np.float64)
        if assume_sorted:
            locations = self._sorted_length_locations(values)
            for index, location in zip(
                np.ndindex(values.shape), locations, strict=True
            ):
                result[index] = self._span_knots[
                    location.span_id
                ] + location.local_u * (
                    self._span_knots[location.span_id + 1]
                    - self._span_knots[location.span_id]
                )
            return self._finish_batch(result, out)
        for index in np.ndindex(values.shape):
            result[index] = self.parameter_at_length(float(values[index]))
        return self._finish_batch(result, out)

    def _sorted_length_locations(
        self, values: NDArray[np.float64]
    ) -> tuple[CurveLocation, ...]:
        flat = values.ravel()
        checked = np.array([self._validate_s(float(value)) for value in flat])
        if np.any(np.diff(checked) < 0.0):
            raise ValueError("assume_sorted=True requires nondecreasing distances")
        locations: list[CurveLocation] = []
        span = 0
        final_span = len(self._spans) - 1
        for target in checked:
            if target == self._total_length:
                locations.append(CurveLocation(final_span, 1.0, self._version))
                continue
            normalized = target / self._scale
            while span < final_span and normalized >= self._prefix_normalized[span + 1]:
                span += 1
            local_target = normalized - self._prefix_normalized[span]
            local = self._spans[span].invert_arc_length(local_target, self._inverse)
            locations.append(CurveLocation(span, local, self._version))
        return tuple(locations)

    # -- stable handles and transactional editing ------------------------

    def point_handle(self, index: int) -> PointHandle:
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise TypeError("index must be an integer")
        integer = int(index)
        if not 0 <= integer < self.num_points:
            raise IndexError("point index out of range")
        return self._handles[integer]

    def index_of(self, handle: PointHandle) -> int:
        if not isinstance(handle, PointHandle):
            raise TypeError("handle must be PointHandle")
        try:
            return self._handles.index(handle)
        except ValueError as exc:
            raise StaleHandleError(
                "PointHandle no longer identifies a live point", point_id=handle.id
            ) from exc

    def _resolve_point(self, point: int | PointHandle) -> int:
        if isinstance(point, PointHandle):
            return self.index_of(point)
        if isinstance(point, (bool, np.bool_)) or not isinstance(
            point, (int, np.integer)
        ):
            raise TypeError("point must be an integer index or PointHandle")
        index = int(point)
        if not 0 <= index < self.num_points:
            raise IndexError("point index out of range")
        return index

    def _validate_repair(self, repair: EditRepair | None) -> EditRepair:
        selected = self._editing.default_repair if repair is None else repair
        if selected not in ("strict_local", "expand", "global"):
            raise ValueError("repair must be 'strict_local', 'expand', or 'global'")
        return selected

    def _commit_edit(
        self,
        *,
        operation: str,
        points: NDArray[np.float64],
        handles: tuple[PointHandle, ...],
        affected_ids: tuple[int, ...],
        repair: EditRepair,
    ) -> EditReport:
        before = self._version
        try:
            state = self._build(
                points,
                handles=handles,
                reuse=repair != "global",
            )
        except PHBSplineValueError:
            raise
        except Exception as exc:
            if repair == "strict_local":
                raise LocalEditFailure(
                    "Strict-local PHBSpline edit could not produce a verified curve",
                    operation=operation,
                    value=str(exc),
                ) from exc
            raise
        old_arrays = {id(span.preimage): span.span_id for span in self._spans}
        affected_spans = tuple(
            i
            for i, span in enumerate(state.spans)
            if id(span.preimage) not in old_arrays
        )
        rebuilt_count = len(affected_spans)
        strict_limit = self._editing.initial_patch_spans
        if strict_limit is None:
            # One midpoint knot gives two compiled PH spans per logical
            # interpolation interval.  A degree-rho patch uses rho intervals
            # to satisfy its two boundary jets plus three guard intervals for
            # shape freedom and regularity margin.
            strict_limit = 2 * (state.preimage_degree + 3)
        allowed = (
            strict_limit if repair == "strict_local" else self._editing.max_patch_spans
        )
        if repair != "global" and rebuilt_count > allowed:
            raise LocalEditFailure(
                "PHBSpline edit exceeded the configured local patch",
                operation=operation,
                quantity="rebuilt span count",
                value=rebuilt_count,
                bound=f"<= {allowed}",
            )
        self._handles = handles
        self._publish(state)
        self._version += 1
        if repair == "global":
            affected_spans = tuple(range(len(state.spans)))
            rebuilt_count = len(state.spans)
        report = EditReport(
            operation=operation,
            version_before=before,
            version_after=self._version,
            affected_point_ids=affected_ids,
            affected_span_ids=affected_spans,
            rebuilt_span_count=rebuilt_count,
            patch_span_count=rebuilt_count,
            iterations=state.diagnostics.iterations,
            refinement_rounds=state.diagnostics.refinement_rounds,
            hidden_spans_added=state.diagnostics.hidden_span_count,
            max_interpolation_residual=state.diagnostics.max_interpolation_residual,
            max_continuity_residual=state.diagnostics.max_continuity_residual,
            min_regularity_ratio=state.diagnostics.min_regularity_ratio,
        )
        self._last_edit_report = report
        return report

    def move_point(
        self,
        point: int | PointHandle,
        value: ArrayLike,
        *,
        repair: EditRepair | None = None,
    ) -> EditReport:
        index = self._resolve_point(point)
        selected = self._validate_repair(repair)
        points = np.array(self._points, copy=True)
        candidate = _edit_point(value)
        points[index] = candidate
        return self._commit_edit(
            operation="move",
            points=points,
            handles=self._handles,
            affected_ids=(self._handles[index].id,),
            repair=selected,
        )

    def insert_point(
        self,
        index: int,
        value: ArrayLike,
        *,
        repair: EditRepair | None = None,
    ) -> InsertResult:
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise TypeError("index must be an integer")
        integer = int(index)
        if integer < 0:
            integer = max(0, self.num_points + integer)
        elif integer > self.num_points:
            integer = self.num_points
        selected = self._validate_repair(repair)
        candidate = _edit_point(value)
        points = np.insert(np.asarray(self._points), integer, candidate, axis=0)
        handle = PointHandle(self._next_handle_id)
        handles = self._handles[:integer] + (handle,) + self._handles[integer:]
        report = self._commit_edit(
            operation="insert",
            points=points,
            handles=handles,
            affected_ids=(handle.id,),
            repair=selected,
        )
        self._next_handle_id += 1
        return InsertResult(handle, report)

    def delete_point(
        self,
        point: int | PointHandle,
        *,
        repair: EditRepair | None = None,
    ) -> EditReport:
        index = self._resolve_point(point)
        selected = self._validate_repair(repair)
        handle = self._handles[index]
        points = np.delete(np.asarray(self._points), index, axis=0)
        handles = self._handles[:index] + self._handles[index + 1 :]
        return self._commit_edit(
            operation="delete",
            points=points,
            handles=handles,
            affected_ids=(handle.id,),
            repair=selected,
        )

    def append_point(
        self, value: ArrayLike, *, repair: EditRepair | None = None
    ) -> InsertResult:
        return self.insert_point(self.num_points, value, repair=repair)

    def prepend_point(
        self, value: ArrayLike, *, repair: EditRepair | None = None
    ) -> InsertResult:
        return self.insert_point(0, value, repair=repair)

    def edit(self, *, repair: EditRepair | None = None) -> PHBSplineEditTransaction:
        return PHBSplineEditTransaction(self, self._validate_repair(repair))

    def snapshot(self) -> PHBSplineSnapshot:
        return PHBSplineSnapshot(self)


class PHBSplineOpen(PHBSpline):
    """Mutable open point-interpolating polynomial PH B-spline."""

    def __init__(
        self,
        points: ArrayLike,
        *,
        g_order: int | None = None,
        c_order: int | None = None,
        curvature_order: int | None = None,
        construction: ConstructionPolicy | None = None,
        editing: EditingPolicy | None = None,
        inverse: InversePolicy | None = None,
        numerics: NumericalPolicy | None = None,
    ) -> None:
        super().__init__(
            points,
            _closed=False,
            g_order=g_order,
            c_order=c_order,
            curvature_order=curvature_order,
            construction=construction,
            editing=editing,
            inverse=inverse,
            numerics=numerics,
        )

    @property
    def closed(self) -> bool:
        return False


class PHBSplineClosed(PHBSpline):
    """Mutable closed point-interpolating polynomial PH B-spline."""

    def __init__(
        self,
        points: ArrayLike,
        *,
        g_order: int | None = None,
        c_order: int | None = None,
        curvature_order: int | None = None,
        construction: ConstructionPolicy | None = None,
        editing: EditingPolicy | None = None,
        inverse: InversePolicy | None = None,
        numerics: NumericalPolicy | None = None,
    ) -> None:
        super().__init__(
            points,
            _closed=True,
            g_order=g_order,
            c_order=c_order,
            curvature_order=curvature_order,
            construction=construction,
            editing=editing,
            inverse=inverse,
            numerics=numerics,
        )

    @property
    def closed(self) -> bool:
        return True

    # -- closed-topology signed area (ClosedSpline_Area_Specification) ----

    @property
    def signed_area(self) -> float:
        """Winding-weighted algebraic area ``(1/2) oint (x dy - y dx)``.

        Correctly rounded analytic Bernstein coefficient sum over the
        committed normalized span positions with cyclic residual join
        closure; counterclockwise traversal is positive.  Lazy per
        committed version: the first query after an edit recomputes only
        the replaced span contributions and O(1) join terms, and repeated
        queries on one version return the cached scalar.
        """
        version = self._version
        state = self._state
        cached = self.__dict__.get("_area_cache")
        if cached is not None and cached[0] == version:
            return cached[1]
        value = self._compute_signed_area(state)
        if self._version == version and self._state is state:
            self.__dict__["_area_cache"] = (version, value)
        return value

    @property
    def area(self) -> float:
        """Nonnegative magnitude ``abs(signed_area)``."""
        return abs(self.signed_area)

    @property
    def fill_area(self) -> float:
        """Nonzero-winding fill area of the enclosed region.

        The Lebesgue measure of the set of points the curve winds about a
        nonzero number of times: the "physical" enclosed area of a
        self-intersecting cycle.  A curve certified free of
        self-intersections returns bitwise ``area``; otherwise the locus
        is decomposed at its certified transversal crossings
        (``ClosedSpline_FillArea_Specification.md``).  Lazy per committed
        version, O(1) on repeated queries.
        """
        version = self._version
        state = self._state
        cached = self.__dict__.get("_fill_area_cache")
        if cached is not None and cached[0] == version:
            return cached[1]
        area_value = self.area
        value = fill_area_source(
            [
                np.column_stack((span.position.real, span.position.imag))
                for span in state.spans
            ],
            [span.preimage for span in state.spans],
            [span.parameter_width for span in state.spans],
            float(state.scale),
            area_value,
        )
        if self._version == version and self._state is state:
            self.__dict__["_fill_area_cache"] = (version, value)
        return value

    def _compute_signed_area(self, state: PHBSplineBuildState) -> float:
        """Area of one captured state with per-span contribution reuse.

        The private per-span cache is keyed by position-array identity plus
        the normalization scale; entries hold a strong array reference and
        are confirmed by object identity, so a recycled object id can never
        alias.  The cache dictionary itself is copy-on-write: a fresh dict
        is swapped in atomically and never mutated in place, which keeps
        snapshot sharing safe (area spec 7.3 / 7.4).
        """
        scale = float(state.scale)
        old_cache = self.__dict__.get("_area_span_cache")
        arrays = []
        balls = []
        new_cache: dict[int, tuple] = {}
        for span in state.spans:
            position = span.position
            array = np.column_stack((position.real, position.imag))
            arrays.append(array)
            entry = None if old_cache is None else old_cache.get(id(position))
            if (
                entry is not None
                and entry[0] is position
                and entry[1] == scale
            ):
                _area_statistics["span_reused"] += 1
            else:
                entry = (position, scale, span_contribution_ball(array))
            if entry[2] is not None:
                new_cache[id(position)] = entry
            balls.append(entry[2])
        self.__dict__["_area_span_cache"] = new_cache
        return source_signed_area(arrays, scale, span_balls=balls)

    def offset(self, distance: numbers.Real) -> ClosedNURBSHandle:
        """Exact closed parallel offset (see :meth:`PHBSpline.offset`).

        The verified closed topology publishes the
        :class:`~ph_spline.nurbs.ClosedNURBSHandle` subtype, which adds the
        closed-only ``signed_area`` and ``area`` properties.
        """
        handle = super().offset(distance)
        assert isinstance(handle, ClosedNURBSHandle)
        return handle

    def snapshot(self) -> PHBSplineClosedSnapshot:
        return PHBSplineClosedSnapshot(self)


class PHBSplineEditTransaction(AbstractContextManager["PHBSplineEditTransaction"]):
    """Draft multiple point edits and publish them with one verified rebuild."""

    def __init__(self, curve: PHBSpline, repair: EditRepair) -> None:
        self._curve = curve
        self._repair = repair
        self._points = [list(row) for row in curve._points]
        self._handles = list(curve._handles)
        self._next_id = curve._next_handle_id
        self._affected_ids: set[int] = set()
        self._active = True

    def _require_active(self) -> None:
        if not self._active:
            raise TransactionError("Edit transaction is no longer active")

    def move_point(self, point: int | PointHandle, value: ArrayLike) -> None:
        self._require_active()
        if isinstance(point, PointHandle):
            try:
                index = self._handles.index(point)
            except ValueError as exc:
                raise StaleHandleError(
                    "PointHandle no longer identifies a transaction point",
                    point_id=point.id,
                ) from exc
        else:
            if isinstance(point, (bool, np.bool_)) or not isinstance(
                point, (int, np.integer)
            ):
                raise TypeError("point must be an integer index or PointHandle")
            index = int(point)
            if not 0 <= index < len(self._points):
                raise IndexError("point index out of range")
        candidate = _edit_point(value)
        self._points[index] = [candidate[0], candidate[1]]
        self._affected_ids.add(self._handles[index].id)

    def insert_point(self, index: int, value: ArrayLike) -> PointHandle:
        self._require_active()
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise TypeError("index must be an integer")
        candidate = _edit_point(value)
        handle = PointHandle(self._next_id)
        self._next_id += 1
        self._points.insert(index, [candidate[0], candidate[1]])
        self._handles.insert(index, handle)
        self._affected_ids.add(handle.id)
        return handle

    def delete_point(self, point: int | PointHandle) -> None:
        self._require_active()
        if isinstance(point, PointHandle):
            try:
                index = self._handles.index(point)
            except ValueError as exc:
                raise StaleHandleError(
                    "PointHandle no longer identifies a transaction point",
                    point_id=point.id,
                ) from exc
        else:
            if isinstance(point, (bool, np.bool_)) or not isinstance(
                point, (int, np.integer)
            ):
                raise TypeError("point must be an integer index or PointHandle")
            index = int(point)
            if not 0 <= index < len(self._points):
                raise IndexError("point index out of range")
        del self._points[index]
        removed = self._handles.pop(index)
        self._affected_ids.add(removed.id)

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._require_active()
        self._active = False
        if exc_type is not None:
            return False
        report = self._curve._commit_edit(
            operation="transaction",
            points=np.asarray(self._points),
            handles=tuple(self._handles),
            affected_ids=tuple(sorted(self._affected_ids)),
            repair=self._repair,
        )
        self._curve._next_handle_id = self._next_id
        self.report = report
        return False


class PHBSplineSnapshot:
    """Immutable query-only view retaining one verified compiled state."""

    _QUERY_NAMES: ClassVar[set[str]] = {
        "arc_length",
        "advance_by_length",
        "closed",
        "curvature_derivative",
        "curvature_vector",
        "curvature_vector_jet",
        "curvature_vectors_at",
        "degree",
        "derivative",
        "derivatives_at",
        "diagnostics",
        "distance_between",
        "frame_at_length",
        "jet",
        "length",
        "length_coordinate",
        "location_at_length",
        "min_curvature_radii",
        "normal",
        "num_points",
        "num_spans",
        "offset",
        "parameter_at_length",
        "parameters_at_length",
        "point",
        "point_at_length",
        "point_after_length",
        "points",
        "points_at",
        "points_at_length",
        "preimage_degree",
        "principal_normal",
        "requested_continuity",
        "signed_curvature",
        "tangent",
        "tangents_at",
        "verified_continuity",
        "version",
    }

    def __init__(self, source: PHBSpline) -> None:
        frozen = object.__new__(type(source))
        frozen.__dict__ = dict(source.__dict__)
        self._curve = frozen

    def __getattr__(self, name: str):
        if name not in self._QUERY_NAMES:
            raise AttributeError(f"PHBSplineSnapshot has no query attribute {name!r}")
        return getattr(self._curve, name)


class PHBSplineClosedSnapshot(PHBSplineSnapshot):
    """Closed snapshot with the closed-topology area interface.

    ``PHBSplineClosed.snapshot()`` returns this subtype, so only closed
    snapshots expose ``signed_area`` and ``area``.  The captured frozen
    state answers area queries against its own version forever; later
    source edits never change it.  Cache sharing with the source is safe
    because every area cache update is a copy-on-write binding swap into
    the owner's own ``__dict__`` (area spec 7.4).
    """

    @property
    def signed_area(self) -> float:
        """Signed area of the captured version (see ``PHBSplineClosed``)."""
        return self._curve.signed_area

    @property
    def area(self) -> float:
        """Nonnegative magnitude ``abs(signed_area)``."""
        return self._curve.area

    @property
    def fill_area(self) -> float:
        """Fill area of the captured version (see ``PHBSplineClosed``)."""
        return self._curve.fill_area


__all__ = [
    "PHBSpline",
    "PHBSplineClosed",
    "PHBSplineClosedSnapshot",
    "PHBSplineEditTransaction",
    "PHBSplineOpen",
    "PHBSplineSnapshot",
]
