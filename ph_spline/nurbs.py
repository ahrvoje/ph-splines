"""Shared immutable NURBS handle and exact PH offset construction.

This module implements the normative exact-offset pipeline of the cubic
specification (section 11.7) and the PH B-spline specification (section
15.6):

- exact homogeneous Bernstein products for ``sigma * r + d * R_L(v)`` and
  ``sigma`` on every compiled source span;
- deterministic positive-weight midpoint refinement;
- canonical projectively-scaled segmented NURBS assembly;
- an independent verification pass (structure, coefficients, breakpoints,
  interior oracle values) that must succeed before a handle is published;
- the read-only :class:`NURBSHandle` with homogeneous de Boor evaluation
  and exact rational frame and curvature queries (spec 11.7.7 / 15.6.9).

Every quantity is produced by finite Bernstein coefficient products - never
by sampling, interpolation, or least-squares fitting.  A handle either
passes the complete verification or :class:`OffsetConstructionError` is
raised; a partially verified handle cannot be constructed because the
constructor requires a module-private proof token issued only by
:func:`build_offset_handle`.
"""

from __future__ import annotations

import math
import numbers
from math import comb, fsum

import numpy as np
from numpy.typing import NDArray

from ph_spline.area import OffsetAreaProvenance, offset_signed_area
from ph_spline.fill_area import fill_area_offset
from ph_spline.exceptions import (
    ArcLengthOutOfRangeError,
    NumericalPrecisionError,
    OffsetConstructionError,
    ParameterOutOfRangeError,
    UndefinedPrincipalNormalError,
    UndefinedTangentError,
)
from ph_spline.offset_metric import build_offset_metric, rebuild_offset_metric

__all__ = [
    "ClosedNURBSHandle",
    "NURBSHandle",
    "build_offset_handle",
    "validate_offset_distance",
]

_EPS = np.finfo(np.float64).eps

#: Fixed maximum midpoint-subdivision depth of the binary64 reference profile.
_MAX_REFINEMENT_DEPTH = 24

#: Ulp slack for clamping ``u`` marginally outside ``[0, 1]`` (spec 15.1).
_ULP_SLACK = 4.0

#: Module-private construction proof token.  ``NURBSHandle`` refuses any
#: other value, so a handle can exist only after the full verification in
#: :func:`build_offset_handle` has succeeded.  This is a structural
#: guarantee, not a convention.
_VERIFIED = object()


def _gamma(n: int) -> float:
    """Standard rounding-accumulation factor ``n * eps / (1 - n * eps)``."""
    x = n * _EPS
    return x / (1.0 - x)


def validate_offset_distance(value: object) -> float:
    """Validate an ``offset(distance)`` argument (spec 11.7.1 / 15.6.1).

    Accept Python and NumPy real scalars; reject Booleans, arrays,
    sequences, NaN, and infinities.  Every finite value, including zero and
    negative values, is valid.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("distance must be a real scalar, not a Boolean")
    if not isinstance(value, numbers.Real):
        raise TypeError(
            f"distance must be a real scalar, not {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise OffsetConstructionError(
            "Offset distance must be finite",
            operation="offset",
            quantity="distance",
            value=result,
            distance=result,
        )
    return result


# ---------------------------------------------------------------------------
# Homogeneous Bernstein offset products (spec 11.7.3 / 15.6.3)
# ---------------------------------------------------------------------------


def _offset_patch(
    controls: NDArray[np.float64],
    speed: NDArray[np.float64],
    d_hat: float,
    span_id: int,
    distance: float,
) -> NDArray[np.float64]:
    """Homogeneous degree-``q`` controls ``(W_k, X_k, Y_k)`` for one span.

    ``controls`` are the degree-``p`` position controls ``C_j``, ``speed``
    the degree-``p-1`` speed coefficients ``s_i``, and ``d_hat`` the
    normalized signed distance.  The derivative controls are
    ``V_i = p * (C_{i+1} - C_i)`` and the left rotation is
    ``R_L(x, y) = (-y, x)``.  Each coefficient is an exact finite integer-
    weighted Bernstein product accumulated with ``math.fsum`` and divided
    once by ``binom(q, k)``.
    """
    p = controls.shape[0] - 1
    q = 2 * p - 1
    v = p * np.diff(controls, axis=0)
    result = np.empty((q + 1, 3), dtype=np.float64)
    for k in range(q + 1):
        i_lo = max(0, k - p)
        i_hi = min(p - 1, k)
        w_terms = []
        x_terms = []
        y_terms = []
        for i in range(i_lo, i_hi + 1):
            j = k - i
            weight = float(comb(p - 1, i) * comb(p, j))
            s_i = float(speed[i])
            w_terms.append(weight * s_i)
            x_terms.append(weight * (s_i * controls[j, 0] - d_hat * v[i, 1]))
            y_terms.append(weight * (s_i * controls[j, 1] + d_hat * v[i, 0]))
        scale = float(comb(q, k))
        result[k, 0] = fsum(w_terms) / scale
        result[k, 1] = fsum(x_terms) / scale
        result[k, 2] = fsum(y_terms) / scale
    if not np.all(np.isfinite(result)):
        raise OffsetConstructionError(
            "A homogeneous offset coefficient is not representable",
            operation="offset",
            span_id=span_id,
            quantity="homogeneous Bernstein coefficient",
            distance=distance,
        )
    return result


def _verify_patch_coefficients(
    patch: NDArray[np.float64],
    controls: NDArray[np.float64],
    speed: NDArray[np.float64],
    d_hat: float,
    span_id: int,
    distance: float,
) -> None:
    """Recompute every coefficient through a second summation path.

    The check path builds the same finite sums with NumPy pairwise
    accumulation in reversed term order, so a coding or accumulation defect
    in the production path cannot cancel identically here.
    """
    p = controls.shape[0] - 1
    q = 2 * p - 1
    v = p * (controls[1:] - controls[:-1])
    rotated = np.empty_like(v)
    rotated[:, 0] = -v[:, 1]
    rotated[:, 1] = v[:, 0]
    magnitude = float(np.max(np.abs(patch)))
    tolerance = _gamma(32 * (q + 1)) * max(1.0, magnitude, abs(d_hat))
    for k in range(q + 1):
        i_lo = max(0, k - p)
        i_hi = min(p - 1, k)
        indices = np.arange(i_hi, i_lo - 1, -1)
        weights = np.array(
            [float(comb(p - 1, i) * comb(p, k - i)) for i in indices]
        )
        s_part = weights * speed[indices]
        w_check = float(np.sum(s_part)) / comb(q, k)
        xy_check = (
            np.sum(
                s_part[:, None] * controls[k - indices]
                + weights[:, None] * (d_hat * rotated[indices]),
                axis=0,
            )
            / comb(q, k)
        )
        gaps = (
            float(abs(patch[k, 0] - w_check)),
            float(abs(patch[k, 1] - xy_check[0])),
            float(abs(patch[k, 2] - xy_check[1])),
        )
        limit = tolerance
        if max(gaps) > limit:
            raise OffsetConstructionError(
                "Independent recomputation of a homogeneous offset "
                "coefficient disagrees with the production value",
                operation="offset",
                span_id=span_id,
                index=k,
                quantity="homogeneous coefficient residual",
                value=max(gaps),
                bound=f"<= {limit:.3e}",
                distance=distance,
            )


# ---------------------------------------------------------------------------
# Positive-weight refinement (spec 11.7.4 / 15.6.4)
# ---------------------------------------------------------------------------


def _split_patch_half(
    patch: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """De Casteljau subdivision of all homogeneous coordinates at ``1/2``."""
    n = patch.shape[0]
    work = np.array(patch, copy=True)
    left = np.empty_like(patch)
    right = np.empty_like(patch)
    left[0] = work[0]
    right[n - 1] = work[n - 1]
    for level in range(1, n):
        work = 0.5 * (work[:-1] + work[1:])
        left[level] = work[0]
        right[n - 1 - level] = work[-1]
    return left, right


def _weights_positive(patch: NDArray[np.float64]) -> bool:
    """Deterministic acceptance test ``W_k > tau_W`` for one leaf patch."""
    weights = patch[:, 0]
    tau = _gamma(32 * patch.shape[0]) * float(np.max(np.abs(weights)))
    return bool(np.all(weights > tau))


def _refine_positive(
    patch: NDArray[np.float64],
    lo: float,
    hi: float,
    span_id: int,
    distance: float,
) -> list[tuple[float, float, NDArray[np.float64]]]:
    """Midpoint-refine one source patch until every leaf weight is positive.

    Left children are processed before right children, the split parameter
    is the exact local midpoint, and the acceptance test depends only on
    the denominator coefficients, so the refinement is identical for every
    distance (the denominator is independent of ``d``).
    """
    leaves: list[tuple[float, float, NDArray[np.float64]]] = []
    stack: list[tuple[float, float, NDArray[np.float64], int]] = [
        (lo, hi, patch, 0)
    ]
    while stack:
        a, b, data, depth = stack.pop()
        if _weights_positive(data):
            leaves.append((a, b, data))
            continue
        if depth >= _MAX_REFINEMENT_DEPTH:
            raise OffsetConstructionError(
                "Positive-weight refinement could not certify the "
                "denominator sign within the fixed subdivision depth",
                operation="offset",
                span_id=span_id,
                quantity="min Bernstein weight",
                value=float(np.min(data[:, 0])),
                bound="> tau_W",
                distance=distance,
                refinement_depth=depth,
            )
        mid = a + 0.5 * (b - a)
        if not (a < mid < b):
            raise OffsetConstructionError(
                "Positive-weight refinement exhausted binary64 breakpoints",
                operation="offset",
                span_id=span_id,
                quantity="breakpoint width",
                value=b - a,
                distance=distance,
                refinement_depth=depth,
            )
        left, right = _split_patch_half(data)
        stack.append((mid, b, right, depth + 1))
        stack.append((a, mid, left, depth + 1))
    return leaves


# ---------------------------------------------------------------------------
# Read-only handle
# ---------------------------------------------------------------------------


class NURBSHandle:
    """Immutable read-only rational spline snapshot with point queries.

    Instances are produced exclusively by the verified exact-offset
    construction; the constructor rejects any call that does not carry the
    module-private proof token.  The handle keeps no reference to the
    source spline and remains valid and unchanged after later source edits.
    """

    __slots__ = (
        "_closed",
        "_degree",
        "_frozen",
        "_homogeneous",
        "_knots",
        "_metric",
        "_num_spans",
        "_points",
        "_weights",
    )

    def __init__(
        self,
        *,
        degree: int,
        knots: NDArray[np.float64],
        control_points: NDArray[np.float64],
        weights: NDArray[np.float64],
        num_spans: int,
        closed: bool,
        metric=None,
        _token: object = None,
    ) -> None:
        if _token is not _VERIFIED:
            raise TypeError(
                "NURBSHandle cannot be constructed directly; use the "
                "offset() method of a PH spline"
            )
        object.__setattr__(self, "_frozen", False)
        self._degree = int(degree)
        self._num_spans = int(num_spans)
        self._closed = bool(closed)
        self._metric = metric
        knots_arr = np.array(knots, dtype=np.float64, copy=True)
        points_arr = np.array(control_points, dtype=np.float64, copy=True)
        weights_arr = np.array(weights, dtype=np.float64, copy=True)
        homogeneous = np.empty((points_arr.shape[0], 3), dtype=np.float64)
        homogeneous[:, 0] = weights_arr * points_arr[:, 0]
        homogeneous[:, 1] = weights_arr * points_arr[:, 1]
        homogeneous[:, 2] = weights_arr
        for array in (knots_arr, points_arr, weights_arr, homogeneous):
            array.setflags(write=False)
        self._knots = knots_arr
        self._points = points_arr
        self._weights = weights_arr
        self._homogeneous = homogeneous
        _verify_structure(
            degree=self._degree,
            knots=knots_arr,
            control_points=points_arr,
            weights=weights_arr,
            num_spans=self._num_spans,
        )
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"{type(self).__name__} is immutable; cannot set {name!r}"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"{type(self).__name__} is immutable; cannot delete {name!r}"
        )

    # -- copy and pickle protocols ----------------------------------------
    #
    # Handles are cached and shipped across process boundaries in ordinary
    # applications.  Restoration bypasses ``__init__`` (and its proof
    # token), so it rebuilds the private homogeneous array from the public
    # arrays and re-runs the structural verification; a corrupted payload
    # raises ``OffsetConstructionError`` instead of materializing an
    # inconsistent handle.

    def __getstate__(self) -> dict:
        state = {
            name: getattr(self, name)
            for name in NURBSHandle.__slots__
            if name not in ("_homogeneous", "_metric")
        }
        # The metric certificate is serialized as its captured raw source
        # state; restoration rebuilds and re-verifies the complete
        # certificate (distance spec 6.1).
        state["_metric_state"] = (
            None if self._metric is None else self._metric.state()
        )
        return state

    def __setstate__(self, state: dict) -> None:
        object.__setattr__(self, "_frozen", False)
        for name in ("_degree", "_num_spans", "_closed"):
            object.__setattr__(self, name, state[name])
        arrays = {}
        for name in ("_knots", "_points", "_weights"):
            value = np.asarray(state[name], dtype=np.float64)
            value.setflags(write=False)
            arrays[name] = value
            object.__setattr__(self, name, value)
        homogeneous = np.empty((arrays["_points"].shape[0], 3))
        homogeneous[:, 0] = arrays["_weights"] * arrays["_points"][:, 0]
        homogeneous[:, 1] = arrays["_weights"] * arrays["_points"][:, 1]
        homogeneous[:, 2] = arrays["_weights"]
        homogeneous.setflags(write=False)
        object.__setattr__(self, "_homogeneous", homogeneous)
        metric_state = state.get("_metric_state")
        object.__setattr__(
            self,
            "_metric",
            None if metric_state is None else rebuild_offset_metric(metric_state),
        )
        _verify_structure(
            degree=self._degree,
            knots=self._knots,
            control_points=self._points,
            weights=self._weights,
            num_spans=self._num_spans,
        )
        object.__setattr__(self, "_frozen", True)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(degree={self._degree}, "
            f"{self.num_control_points} controls, {self._num_spans} spans, "
            f"{'closed' if self._closed else 'open'})"
        )

    # -- inspection -------------------------------------------------------

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def knots(self) -> NDArray[np.float64]:
        return self._knots

    @property
    def control_points(self) -> NDArray[np.float64]:
        return self._points

    @property
    def weights(self) -> NDArray[np.float64]:
        return self._weights

    @property
    def num_control_points(self) -> int:
        return int(self._points.shape[0])

    @property
    def num_spans(self) -> int:
        return self._num_spans

    @property
    def domain(self) -> tuple[float, float]:
        return (0.0, 1.0)

    @property
    def closed(self) -> bool:
        return self._closed

    # -- evaluation -------------------------------------------------------

    def _validate_u(self, value: object) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("u must be a real scalar, not a Boolean")
        if not isinstance(value, numbers.Real):
            raise TypeError(
                f"u must be a real scalar, not {type(value).__name__}"
            )
        u = float(value)
        if math.isnan(u):
            raise ParameterOutOfRangeError(
                "Parameter u must not be NaN", quantity="u", value=u
            )
        if u < 0.0:
            if u >= -_ULP_SLACK * _EPS:
                return 0.0
            raise ParameterOutOfRangeError(
                "Parameter u is below the domain [0, 1]",
                quantity="u",
                value=u,
                bound=">= 0",
            )
        if u > 1.0:
            if u <= 1.0 + _ULP_SLACK * _EPS:
                return 1.0
            raise ParameterOutOfRangeError(
                "Parameter u is above the domain [0, 1]",
                quantity="u",
                value=u,
                bound="<= 1",
            )
        return u

    # -- distance queries (OffsetNURBS_Distance_Specification) -----------

    def _require_metric(self):
        metric = self._metric
        if metric is None:
            raise OffsetConstructionError(
                "This handle carries no verified distance certificate; "
                "distance queries are defined only for handles produced "
                "by offset() of a PH spline",
                operation="offset-metric",
                quantity="metric certificate",
            )
        return metric

    def _validate_s(self, value: object) -> float:
        """Validate an arc-length scalar (distance spec 3.5).

        Accepts real scalars in ``[0, length]`` with a four-ulp endpoint
        clamp; NaN, infinities, and material excursions raise
        :class:`ArcLengthOutOfRangeError`.
        """
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("s must be a real scalar, not a Boolean")
        if not isinstance(value, numbers.Real):
            raise TypeError(
                f"s must be a real scalar, not {type(value).__name__}"
            )
        s = float(value)
        if math.isnan(s):
            raise ArcLengthOutOfRangeError(
                "Arc length s must not be NaN", quantity="s", value=s
            )
        total = self._metric.length
        if math.isinf(s):
            raise ArcLengthOutOfRangeError(
                "Arc length s must be finite",
                quantity="s",
                value=s,
                bound=f"[0, {total!r}]",
            )
        slack = _ULP_SLACK * math.ulp(total)
        if s < 0.0:
            if s >= -slack:
                return 0.0
            raise ArcLengthOutOfRangeError(
                "Arc length s is below the domain [0, length]",
                quantity="s",
                value=s,
                bound=">= 0",
            )
        if s > total:
            if s <= total + slack:
                return total
            raise ArcLengthOutOfRangeError(
                "Arc length s is above the domain [0, length]",
                quantity="s",
                value=s,
                bound=f"<= {total!r}",
            )
        return s

    @property
    def length(self) -> float:
        """Total unsigned traversal length of the offset locus.

        Correctly rounded value of the exact-reference length, computed
        during verified construction and returned in O(1).
        """
        return self._require_metric().length

    @property
    def cusps(self) -> tuple:
        """Certified offset cusps as ``OffsetCusp(parameter, multiplicity)``.

        Ascending global parameters of every zero-speed point of the
        offset locus, computed at construction by complete exact-rational
        root isolation of the cusp polynomial and refined to within two
        ulps of the exact stationary parameter.  Odd multiplicity marks a
        direction reversal, even multiplicity a tangential zero-speed
        graze; sub-ulp root pairs coalesce into one record.  Empty for
        cusp-free offsets (in particular for ``d == 0``).  O(1) after
        construction.
        """
        return self._require_metric().cusps

    def arc_length(self, u: object) -> float:
        """Unsigned traversal distance from ``point(0)`` to ``point(u)``.

        The result is the correctly rounded exact-reference value (which
        satisfies the faithful-rounding contract of the distance
        specification, section 11.1) with exact endpoint identities
        ``arc_length(0.0) == 0.0`` and ``arc_length(1.0) == length``.
        """
        metric = self._require_metric()
        return metric.arc_length(self._validate_u(u))

    def parameter_at_length(self, s: object) -> float:
        """Unique parameter ``u`` with ``arc_length(u) == s``.

        Solved from the elementary offset-distance function by a
        safeguarded bracketed Newton iteration with a certified residual
        acceptance gate, falling back to the best-representable-parameter
        search of the distance specification (section 10).
        """
        metric = self._require_metric()
        return metric.parameter_at_length(self._validate_s(s))

    def point_at_length(self, s: object) -> NDArray[np.float64]:
        """Point reached after travelling ``s`` along the offset locus.

        Semantically identical to
        ``point(parameter_at_length(s))``; the inverse and the verified
        rational evaluation share one call chain, so the two-call
        expression returns the same coordinates for the same accepted
        argument.
        """
        metric = self._require_metric()
        return self.point(metric.parameter_at_length(self._validate_s(s)))

    def point(self, u: object) -> NDArray[np.float64]:
        """Rational point by homogeneous de Boor and one final division."""
        value = self._validate_u(u)
        numerator_x, numerator_y, denominator = _deboor_homogeneous(
            self._knots, self._homogeneous, self._degree, value
        )
        if not (denominator > 0.0 and math.isfinite(denominator)):
            raise NumericalPrecisionError(
                "NURBS denominator is not positive and finite",
                quantity="denominator",
                value=denominator,
                bound="> 0",
            )
        result = np.array(
            [numerator_x / denominator, numerator_y / denominator],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(result)):
            raise NumericalPrecisionError(
                "NURBS evaluation produced a nonfinite point",
                quantity="point",
                value=result.tolist(),
            )
        return result

    # -- frame and curvature queries (spec 11.7.7 / 15.6.9) ---------------

    def _rational_jet(
        self, u: float
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
        float,
    ]:
        """Rational point, velocity, and acceleration at an accepted ``u``.

        The canonical segmented knot vector makes every span a pure
        homogeneous Bernstein patch, so the derivatives come from one
        exact de Casteljau jet of the located span followed by the
        quotient rule

        ``C = A / w``,
        ``C' = (A' - C w') / w``,
        ``C'' = (A'' - 2 w' C' - w'' C) / w``,

        in user coordinates and with respect to the global parameter.
        Also returns the deterministic first- and second-derivative
        noise floors ``tau1`` and ``tau2`` of the zero-speed and
        zero-curvature acceptance gates.  Span selection is right-sided,
        exactly as in :meth:`point`.
        """
        q = self._degree
        knots = self._knots
        k = int(np.searchsorted(knots, u, side="right")) - 1
        k = min(max(k, q), self._points.shape[0] - 1)
        span = (k - q) // q
        lo = float(knots[k])
        hi = float(knots[k + 1])
        width = hi - lo
        if not width > 0.0:
            raise NumericalPrecisionError(
                "Offset span breakpoint interval collapsed",
                index=span,
                quantity="span width",
                value=width,
                bound="> 0",
            )
        t = (u - lo) / width
        patch = self._homogeneous[span * q : span * q + q + 1]
        h0, h1, h2 = _decasteljau_jet(patch, t)
        w = float(h0[2])
        if not (w > 0.0 and math.isfinite(w)):
            raise NumericalPrecisionError(
                "NURBS denominator is not positive and finite",
                quantity="denominator",
                value=w,
                bound="> 0",
            )
        point = h0[:2] / w
        w1 = float(h1[2]) / width
        w2 = float(h2[2]) / width**2
        velocity = (h1[:2] / width - w1 * point) / w
        acceleration = (
            h2[:2] / width**2 - 2.0 * w1 * velocity - w2 * point
        ) / w
        condition = (
            float(np.max(np.abs(patch)))
            * (1.0 + float(np.max(np.abs(point))))
            / w
        )
        base = _gamma(32 * (q + 1)) * condition
        tau1 = base * (q / width)
        tau2 = base * (q / width) ** 2
        return point, velocity, acceleration, tau1, tau2

    def _frame_jet(
        self, u: object
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
        float,
        float,
    ]:
        """Validated jet behind every frame query.

        Applies the scalar rules of :meth:`point`, then the deterministic
        zero-speed acceptance gate ``speed > tau1``: below the noise
        floor the traversal direction is not numerically meaningful, so
        every frame query raises :class:`UndefinedTangentError` there —
        in particular at every certified parameter of :attr:`cusps`.
        """
        value = self._validate_u(u)
        point, velocity, acceleration, tau1, tau2 = self._rational_jet(value)
        speed = float(np.hypot(velocity[0], velocity[1]))
        if not (
            math.isfinite(speed)
            and np.all(np.isfinite(acceleration))
        ):
            raise NumericalPrecisionError(
                "Offset derivative evaluation produced a nonfinite value",
                quantity="derivative",
                value=(velocity.tolist(), acceleration.tolist()),
            )
        if not speed > tau1:
            raise UndefinedTangentError(
                "Offset frame is undefined at a zero-speed cusp",
                quantity="offset speed",
                value=speed,
                bound=f"> {tau1:.3e}",
            )
        return point, velocity, acceleration, speed, tau1, tau2

    def tangent(self, u: object) -> NDArray[np.float64]:
        """Unit traversal tangent of the offset locus at ``u``.

        Computed by exact rational differentiation of the published
        homogeneous controls — never by sampling or a source callback.
        Equals ``sign(1 - d * kappa(u)) * T(u)`` of the source in exact
        arithmetic, so the direction reverses between odd-multiplicity
        cusps.  Raises :class:`UndefinedTangentError` at a zero-speed
        cusp, where the direction is undefined.
        """
        _, velocity, _, speed, _, _ = self._frame_jet(u)
        return np.array(
            [velocity[0] / speed, velocity[1] / speed], dtype=np.float64
        )

    def normal(self, u: object, side: str = "left") -> NDArray[np.float64]:
        """Unit left or right normal of the offset traversal at ``u``.

        The left normal is the quarter-turn ``R_L(T) = (-T_y, T_x)`` of
        the unit tangent, the right normal its negation.  Equals
        ``sign(1 - d * kappa(u)) * N(u)`` of the same-side source normal
        in exact arithmetic.
        """
        if side == "left":
            sign = 1.0
        elif side == "right":
            sign = -1.0
        else:
            raise ValueError(
                f'normal() side must be "left" or "right", got {side!r}'
            )
        _, velocity, _, speed, _, _ = self._frame_jet(u)
        tx = velocity[0] / speed
        ty = velocity[1] / speed
        return np.array([-sign * ty, sign * tx], dtype=np.float64)

    def signed_curvature(self, u: object) -> float:
        """Signed curvature of the offset locus at ``u``.

        ``kappa_d = (r_d' x r_d'') / ||r_d'||**3`` with respect to the
        handle's own traversal direction; positive turns left.  Equals
        ``kappa / |1 - d * kappa|`` of the source in exact arithmetic:
        the sign matches the source curvature and the magnitude grows
        without bound toward a cusp.  Finite values are returned as
        computed; a zero-speed cusp raises
        :class:`UndefinedTangentError` and a nonfinite result raises
        :class:`NumericalPrecisionError`.
        """
        _, velocity, acceleration, speed, _, _ = self._frame_jet(u)
        cross = float(
            velocity[0] * acceleration[1] - velocity[1] * acceleration[0]
        )
        result = cross / speed**3
        if not math.isfinite(result):
            raise NumericalPrecisionError(
                "Offset signed curvature is not finite",
                quantity="signed curvature",
                value=result,
            )
        return result

    def curvature_vector(self, u: object) -> NDArray[np.float64]:
        """Curvature vector ``K = kappa_d * N_L`` of the offset at ``u``.

        Parameterization-independent; equals
        ``kappa / (1 - d * kappa) * N_L`` of the source in exact
        arithmetic (no absolute value: the tangent and normal reversals
        cancel).  A straight offset returns the zero vector.
        """
        _, velocity, acceleration, speed, _, _ = self._frame_jet(u)
        cross = float(
            velocity[0] * acceleration[1] - velocity[1] * acceleration[0]
        )
        kappa = cross / speed**3
        if not math.isfinite(kappa):
            raise NumericalPrecisionError(
                "Offset signed curvature is not finite",
                quantity="signed curvature",
                value=kappa,
            )
        tx = velocity[0] / speed
        ty = velocity[1] / speed
        return np.array([-kappa * ty, kappa * tx], dtype=np.float64)

    def principal_normal(self, u: object) -> NDArray[np.float64]:
        """Unit normal toward the offset's local center of curvature.

        ``sign(kappa_d) * N_L(u)``; the offset shares its curvature
        center with the source point it offsets.  Raises
        :class:`UndefinedPrincipalNormalError` when the curvature
        numerator does not exceed its deterministic roundoff noise floor
        (straight offsets), and :class:`UndefinedTangentError` at a
        zero-speed cusp.
        """
        _, velocity, acceleration, speed, tau1, tau2 = self._frame_jet(u)
        cross = float(
            velocity[0] * acceleration[1] - velocity[1] * acceleration[0]
        )
        magnitude = float(np.hypot(acceleration[0], acceleration[1]))
        floor = speed * tau2 + magnitude * tau1
        if not abs(cross) > floor:
            raise UndefinedPrincipalNormalError(
                "Principal normal is undefined at zero offset curvature",
                quantity="curvature numerator",
                value=cross,
                bound=f"abs > {floor:.3e}",
            )
        sign = 1.0 if cross > 0.0 else -1.0
        tx = velocity[0] / speed
        ty = velocity[1] / speed
        return np.array([-sign * ty, sign * tx], dtype=np.float64)


class ClosedNURBSHandle(NURBSHandle):
    """Closed exact-offset handle with the closed-topology area interface.

    ``build_offset_handle`` instantiates this subtype exactly when the
    verified source topology is closed, so only closed offsets carry
    ``signed_area`` and ``area`` (``ClosedSpline_Area_Specification.md``,
    section 3.2).  The handle owns an immutable area provenance record
    (captured source position controls plus the shared exact metric
    certificate); the first area query computes lazily and caches, so
    repeated queries are O(1).
    """

    __slots__ = ("_area_cache", "_area_provenance", "_fill_area_cache")

    def __init__(self, *, area_provenance: OffsetAreaProvenance, **kwargs) -> None:
        object.__setattr__(self, "_area_provenance", area_provenance)
        object.__setattr__(self, "_area_cache", None)
        object.__setattr__(self, "_fill_area_cache", None)
        super().__init__(**kwargs)

    @property
    def signed_area(self) -> float:
        """Winding-weighted algebraic area of the exact offset locus.

        Correctly rounded value of ``A_0 - d L_0 + pi nu d**2`` evaluated
        from the captured exact source state: exact Bernstein source area,
        exact PH source length and the certified integer tangent turning
        number.  Counterclockwise traversal is positive; cusped and
        self-intersecting offsets keep their defined algebraic value.
        Lazy on first query, O(1) afterwards.
        """
        cached = self._area_cache
        if cached is None:
            cached = offset_signed_area(self._area_provenance)
            object.__setattr__(self, "_area_cache", cached)
        return cached

    @property
    def area(self) -> float:
        """Nonnegative magnitude ``abs(signed_area)``."""
        return abs(self.signed_area)

    @property
    def fill_area(self) -> float:
        """Nonzero-winding fill area of the exact offset locus.

        The "physical" enclosed area of the parallel curve: cusp loops
        and self-intersections are decomposed at certified transversal
        crossings of the exact-reference locus, and a cusp-free simple
        offset returns bitwise ``area``
        (``ClosedSpline_FillArea_Specification.md``).  Lazy on first
        query, O(1) afterwards.
        """
        cached = self._fill_area_cache
        if cached is None:
            cached = fill_area_offset(self._area_provenance, self.area)
            object.__setattr__(self, "_fill_area_cache", cached)
        return cached

    # -- copy and pickle protocols ----------------------------------------
    #
    # The area caches are derived state and are omitted; the provenance
    # position controls travel with the handle, while its exact preimage
    # state is shared with (and rebuilt through) the metric certificate.

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state["_area_position_spans"] = tuple(
            np.asarray(array) for array in self._area_provenance.position_spans
        )
        return state

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        try:
            position_spans = state["_area_position_spans"]
        except KeyError:
            raise OffsetConstructionError(
                "Closed offset payload lacks its area provenance",
                operation="offset",
                quantity="area provenance",
            ) from None
        provenance = OffsetAreaProvenance(
            position_spans=position_spans, metric=self._metric
        )
        object.__setattr__(self, "_area_provenance", provenance)
        object.__setattr__(self, "_area_cache", None)
        object.__setattr__(self, "_fill_area_cache", None)


def _deboor_homogeneous(
    knots: NDArray[np.float64],
    homogeneous: NDArray[np.float64],
    degree: int,
    u: float,
) -> tuple[float, float, float]:
    """Standard de Boor recurrence on homogeneous ``(wx, wy, w)`` controls.

    Internal knot selection is right-sided; ``u == 1`` selects the final
    nondegenerate span.
    """
    count = homogeneous.shape[0]
    k = int(np.searchsorted(knots, u, side="right")) - 1
    k = min(max(k, degree), count - 1)
    work = np.array(homogeneous[k - degree : k + 1], copy=True)
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            i = k - degree + j
            left = float(knots[i])
            right = float(knots[i + degree - r + 1])
            width = right - left
            if not width > 0.0:
                raise NumericalPrecisionError(
                    "De Boor knot interval collapsed",
                    index=i,
                    quantity="knot interval",
                    value=width,
                    bound="> 0",
                )
            alpha = (u - left) / width
            work[j] = (1.0 - alpha) * work[j - 1] + alpha * work[j]
    return (
        float(work[degree, 0]),
        float(work[degree, 1]),
        float(work[degree, 2]),
    )


def _decasteljau_jet(
    patch: NDArray[np.float64], t: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Homogeneous value and first two local derivatives at ``t``.

    One de Casteljau pass over a degree-``n`` homogeneous Bernstein patch,
    retaining the last two triangle levels: with the level-``n - 1``
    points ``a0, a1`` and the level-``n - 2`` points ``c0, c1, c2``,

    ``H(t) = (1 - t) a0 + t a1``,
    ``H'(t) = n (a1 - a0)``,
    ``H''(t) = n (n - 1) (c2 - 2 c1 + c0)``.

    Every step is a finite combination of the published controls; no
    sampling or finite differencing is involved.  A degree-1 patch has an
    identically zero second derivative.
    """
    n = patch.shape[0] - 1
    work = np.array(patch, copy=True)
    s = 1.0 - t
    first = np.zeros(patch.shape[1], dtype=np.float64)
    second = np.zeros(patch.shape[1], dtype=np.float64)
    for width in range(n, 0, -1):
        if width == 2:
            second = float(n * (n - 1)) * (
                work[2] - 2.0 * work[1] + work[0]
            )
        if width == 1:
            first = float(n) * (work[1] - work[0])
        work[:width] = s * work[:width] + t * work[1 : width + 1]
    return np.array(work[0], copy=True), first, second


# ---------------------------------------------------------------------------
# Structural verification (spec 11.7.5 / 15.6.7)
# ---------------------------------------------------------------------------


def _verify_structure(
    *,
    degree: int,
    knots: NDArray[np.float64],
    control_points: NDArray[np.float64],
    weights: NDArray[np.float64],
    num_spans: int,
) -> None:
    """Verify every structural array contract of the public handle."""
    count = control_points.shape[0]
    checks = (
        degree >= 1,
        num_spans >= 1,
        count == num_spans * degree + 1,
        knots.shape == (count + degree + 1,),
        control_points.shape == (count, 2),
        weights.shape == (count,),
    )
    if not all(checks):
        raise OffsetConstructionError(
            "Offset NURBS arrays violate their structural contract",
            operation="offset",
            quantity="array shapes",
            value=(degree, knots.shape, control_points.shape, weights.shape),
        )
    if not (
        np.all(np.isfinite(knots))
        and np.all(np.isfinite(control_points))
        and np.all(np.isfinite(weights))
    ):
        raise OffsetConstructionError(
            "Offset NURBS arrays contain a nonfinite value",
            operation="offset",
            quantity="finite arrays",
        )
    if not np.all(weights > 0.0):
        raise OffsetConstructionError(
            "Offset NURBS weights are not strictly positive",
            operation="offset",
            quantity="min weight",
            value=float(np.min(weights)),
            bound="> 0",
        )
    if np.any(np.diff(knots) < 0.0):
        raise OffsetConstructionError(
            "Offset NURBS knots are not nondecreasing",
            operation="offset",
            quantity="diff(knots)",
            bound=">= 0",
        )
    breaks = np.unique(knots)
    expected = np.concatenate(
        (
            np.full(degree + 1, breaks[0]),
            np.repeat(breaks[1:-1], degree),
            np.full(degree + 1, breaks[-1]),
        )
    )
    if not (
        breaks[0] == 0.0
        and breaks[-1] == 1.0
        and breaks.size == num_spans + 1
        and expected.shape == knots.shape
        and np.array_equal(expected, knots)
    ):
        raise OffsetConstructionError(
            "Offset NURBS knot multiplicities violate the canonical "
            "clamped segmented form",
            operation="offset",
            quantity="knot multiplicities",
        )


# ---------------------------------------------------------------------------
# Verified construction driver
# ---------------------------------------------------------------------------


def build_offset_handle(
    *,
    span_controls: list[NDArray[np.float64]],
    span_speeds: list[NDArray[np.float64]],
    span_hodographs: list[NDArray[np.float64]],
    hodograph_tolerance: float,
    breakpoints: NDArray[np.float64],
    distance: float,
    distance_normalized: float,
    origin: tuple[float, float],
    scale: float,
    closed: bool,
    join_tolerance: float,
    oracle,
    metric_preimages: list | None = None,
    metric_widths: list | None = None,
) -> NURBSHandle:
    """Construct and fully verify one exact offset NURBS handle.

    All span data is in the family's normalized frame: ``span_controls[i]``
    holds the degree-``p`` position controls, ``span_speeds[i]`` the
    degree-``p-1`` speed coefficients from the stored PH preimage, and
    ``span_hodographs[i]`` the stored hodograph controls used only for the
    independent derivative check.  ``oracle(u)`` returns the source point
    and left unit normal in user coordinates and must not depend on any
    state that a later source edit can change.

    When ``metric_preimages`` (per-span complex Bernstein preimages) and
    ``metric_widths`` (per-span local parameter-width factors) are given,
    the complete distance metric certificate of
    ``OffsetNURBS_Distance_Specification.md`` is built and verified
    before the handle is published, so geometry and metric succeed or
    fail atomically.  Handles built without metric data (internal test
    pipelines) reject distance queries.

    The function either returns a completely verified handle or raises
    :class:`OffsetConstructionError`; no partial handle can escape.
    """
    d_hat = distance_normalized
    if not math.isfinite(d_hat):
        raise OffsetConstructionError(
            "Normalized offset distance is not representable",
            operation="offset",
            quantity="distance / H",
            value=d_hat,
            distance=distance,
        )
    source_count = len(span_controls)
    if not (
        source_count >= 1
        and len(span_speeds) == source_count
        and len(span_hodographs) == source_count
        and breakpoints.shape == (source_count + 1,)
        and breakpoints[0] == 0.0
        and breakpoints[-1] == 1.0
        and np.all(np.diff(breakpoints) > 0.0)
    ):
        raise OffsetConstructionError(
            "Offset source span data is inconsistent",
            operation="offset",
            quantity="span data sizes",
            distance=distance,
        )
    p = span_controls[0].shape[0] - 1
    q = 2 * p - 1

    # -- per-span products with the independent hodograph check ----------
    refined: list[tuple[float, float, NDArray[np.float64]]] = []
    for index in range(source_count):
        controls = np.asarray(span_controls[index], dtype=np.float64)
        speed = np.asarray(span_speeds[index], dtype=np.float64)
        hodograph = np.asarray(span_hodographs[index], dtype=np.float64)
        if (
            controls.shape != (p + 1, 2)
            or speed.shape != (p,)
            or hodograph.shape != (p, 2)
        ):
            raise OffsetConstructionError(
                "Offset span arrays have inconsistent degrees",
                operation="offset",
                span_id=index,
                quantity="span array shapes",
                distance=distance,
            )
        derivative = p * np.diff(controls, axis=0)
        gap = float(np.max(np.abs(derivative - hodograph)))
        limit = hodograph_tolerance * max(
            1.0, float(np.max(np.abs(hodograph)))
        )
        if gap > limit:
            raise OffsetConstructionError(
                "Span derivative controls disagree with the stored "
                "PH hodograph",
                operation="offset",
                span_id=index,
                quantity="max |p*diff(C) - hodograph|",
                value=gap,
                bound=f"<= {limit:.3e}",
                distance=distance,
            )
        patch = _offset_patch(controls, speed, d_hat, index, distance)
        _verify_patch_coefficients(
            patch, controls, speed, d_hat, index, distance
        )
        refined.extend(
            _refine_positive(
                patch,
                float(breakpoints[index]),
                float(breakpoints[index + 1]),
                index,
                distance,
            )
        )

    # -- join verification and canonical projective assembly -------------
    origin_x, origin_y = float(origin[0]), float(origin[1])

    def _to_user(x: float, y: float) -> tuple[float, float]:
        return (origin_x + scale * x, origin_y + scale * y)

    def _euclidean(row: NDArray[np.float64]) -> tuple[float, float]:
        return (float(row[1]) / float(row[0]), float(row[2]) / float(row[0]))

    def _oracle_user(u: float) -> tuple[float, float]:
        point, normal = oracle(u)
        return (
            float(point[0]) + distance * float(normal[0]),
            float(point[1]) + distance * float(normal[1]),
        )

    def _patch_tolerance(patch: NDArray[np.float64]) -> float:
        """Degree- and scale-aware forward bound in normalized units."""
        numerator = float(np.max(np.abs(patch[:, 1:])))
        w_min = float(np.min(patch[:, 0]))
        condition = (numerator + float(np.max(patch[:, 0]))) / w_min
        return (
            64.0 * (q + 1) * _EPS * max(1.0, condition)
            + 2.0 * join_tolerance * abs(d_hat)
        )

    def _check_value(
        expected: tuple[float, float],
        actual: tuple[float, float],
        tol_normalized: float,
        label: str,
        u: float,
    ) -> None:
        tol_user = scale * tol_normalized + 16.0 * _EPS * (
            abs(origin_x)
            + abs(origin_y)
            + abs(expected[0])
            + abs(expected[1])
        )
        gap = math.hypot(actual[0] - expected[0], actual[1] - expected[1])
        if not gap <= tol_user:
            raise OffsetConstructionError(
                f"Offset verification failed at {label}",
                operation="offset",
                quantity=f"|offset({u:.17g}) - (r + d*N_L)|",
                value=gap,
                bound=f"<= {tol_user:.3e}",
                distance=distance,
            )

    for patch_index, (lo, hi, patch) in enumerate(refined):
        tol = _patch_tolerance(patch)
        for local, u in ((0.0, lo), (1.0, hi)):
            row = patch[0] if local == 0.0 else patch[-1]
            value = _to_user(*_euclidean(row))
            _check_value(
                _oracle_user(u), value, tol, f"breakpoint {u:.17g}", u
            )
        for fraction in (0.25, 0.5, 0.75):
            u = lo + fraction * (hi - lo)
            row = _decasteljau_rows(patch, fraction)
            if not (row[0] > 0.0 and math.isfinite(row[0])):
                raise OffsetConstructionError(
                    "Refined patch denominator is not positive",
                    operation="offset",
                    index=patch_index,
                    quantity="denominator",
                    value=float(row[0]),
                    bound="> 0",
                    distance=distance,
                )
            value = _to_user(*_euclidean(row))
            _check_value(
                _oracle_user(u), value, tol, f"interior parameter {u:.17g}", u
            )

    assembled: list[NDArray[np.float64]] = []
    for patch_index, (lo, hi, patch) in enumerate(refined):
        if not assembled:
            assembled.append(np.array(patch, copy=True))
            continue
        previous = assembled[-1]
        prev_end = _euclidean(previous[-1])
        next_start = _euclidean(patch[0])
        # _patch_tolerance already scales with the homogeneous numerator
        # magnitude, so no further coordinate factor is applied here.
        tol = max(_patch_tolerance(previous), _patch_tolerance(patch))
        gap = math.hypot(
            prev_end[0] - next_start[0], prev_end[1] - next_start[1]
        )
        if not gap <= tol:
            raise OffsetConstructionError(
                "Adjacent offset patches disagree at their shared "
                "breakpoint before projective scaling",
                operation="offset",
                index=patch_index,
                quantity="Euclidean endpoint gap",
                value=gap,
                bound=f"<= {tol:.3e}",
                distance=distance,
            )
        factor = float(previous[-1, 0]) / float(patch[0, 0])
        if not (factor > 0.0 and math.isfinite(factor)):
            raise OffsetConstructionError(
                "Projective joint scale is not representable",
                operation="offset",
                index=patch_index,
                quantity="W-(1) / W+(0)",
                value=factor,
                distance=distance,
            )
        scaled = factor * patch
        if not np.all(np.isfinite(scaled)):
            raise OffsetConstructionError(
                "Projective rescaling overflowed a homogeneous control",
                operation="offset",
                index=patch_index,
                quantity="scaled homogeneous controls",
                distance=distance,
            )
        scaled[0] = previous[-1]
        assembled.append(scaled)

    # -- canonical knots, user-coordinate controls, publication ----------
    final_breaks = np.array(
        [interval[0] for interval in refined] + [refined[-1][1]],
        dtype=np.float64,
    )
    span_count = len(assembled)
    knots = np.concatenate(
        (
            np.zeros(q + 1),
            np.repeat(final_breaks[1:-1], q),
            np.ones(q + 1),
        )
    )
    homogeneous = np.concatenate(
        [assembled[0]] + [patch[1:] for patch in assembled[1:]], axis=0
    )
    weights = homogeneous[:, 0]
    controls_user = np.empty((homogeneous.shape[0], 2), dtype=np.float64)
    controls_user[:, 0] = origin_x + scale * (homogeneous[:, 1] / weights)
    controls_user[:, 1] = origin_y + scale * (homogeneous[:, 2] / weights)
    if not np.all(np.isfinite(controls_user)):
        raise OffsetConstructionError(
            "An offset control point is not representable in user "
            "coordinates",
            operation="offset",
            quantity="control points",
            distance=distance,
        )

    # -- distance metric certificate (atomic with geometry) --------------
    metric = None
    if metric_preimages is not None:
        metric = build_offset_metric(
            span_preimages=metric_preimages,
            span_widths=metric_widths,
            breakpoints=[float(b) for b in breakpoints],
            distance=distance,
            scale=scale,
            closed=closed,
        )

    handle_kwargs = dict(
        degree=q,
        knots=knots,
        control_points=controls_user,
        weights=weights,
        num_spans=span_count,
        closed=closed,
        metric=metric,
        _token=_VERIFIED,
    )
    if closed and metric is not None:
        # Closed verified topology: publish the closed subtype with its
        # immutable area provenance (capture only; no area is evaluated).
        handle = ClosedNURBSHandle(
            area_provenance=OffsetAreaProvenance(
                position_spans=span_controls, metric=metric
            ),
            **handle_kwargs,
        )
    else:
        handle = NURBSHandle(**handle_kwargs)

    # -- independent post-publication checks through the public path -----
    for u in final_breaks:
        value = handle.point(float(u))
        expected = _oracle_user(float(u))
        index = int(np.searchsorted(final_breaks, u, side="right")) - 1
        index = min(max(index, 0), span_count - 1)
        tol = _patch_tolerance(refined[index][2])
        _check_value(
            expected,
            (float(value[0]), float(value[1])),
            tol,
            f"published knot {float(u):.17g}",
            float(u),
        )
    if closed:
        start = handle.point(0.0)
        end = handle.point(1.0)
        tol = scale * (
            max(
                _patch_tolerance(refined[0][2]),
                _patch_tolerance(refined[-1][2]),
            )
        ) + 16.0 * _EPS * float(np.max(np.abs(start)))
        seam_gap = float(np.hypot(*(end - start)))
        if not seam_gap <= tol:
            raise OffsetConstructionError(
                "Closed offset seam values disagree",
                operation="offset",
                quantity="|offset(0) - offset(1)|",
                value=seam_gap,
                bound=f"<= {tol:.3e}",
                distance=distance,
            )
    return handle


def _decasteljau_rows(
    patch: NDArray[np.float64], t: float
) -> NDArray[np.float64]:
    """Plain de Casteljau on a ``(n, 3)`` homogeneous Bernstein patch."""
    work = np.array(patch, copy=True)
    s = 1.0 - t
    for width in range(work.shape[0] - 1, 0, -1):
        work[:width] = s * work[:width] + t * work[1 : width + 1]
    return work[0]
